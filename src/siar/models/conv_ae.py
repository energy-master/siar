# Vixen Intelligence c.2026
"""``conv_ae`` — a convolutional autoencoder that flags what it cannot rebuild.

Train it to reconstruct patches of a spectrogram drawn from a corpus you believe is mostly
normal. It gets good at rebuilding the textures it saw often, and stays bad at rebuilding
anything else. The squared residual, kept at **per-pixel** resolution, is the anomaly signal.

**The bottleneck has to be narrow AND local. Getting either half wrong breaks the detector, in
opposite directions.**

*Too wide* and the autoencoder learns the identity map. It reconstructs anomalies perfectly, the
error map goes flat, and the detector finds nothing — while scoring *better and better* on
reconstruction loss as it gets worse. This trap is easy to fall into unnoticed: two stride-2
convs to 32 channels on a 32x64 patch gives 32 x 8 x 16 = 4096 latent units for a 2048-pixel
input. That is an *expansion*, not a bottleneck, and it is a real bug in a real acoustic AE.
Hence :data:`MAX_LATENT_FRACTION` and the hard check in :func:`validate_config`.

*Too global* and the detector still "works" but loses the ability to say **where**. A ``Linear``
bottleneck connects every latent unit to every pixel, so an anomaly anywhere in the patch
corrupts the entire reconstruction and the error smears across the whole frequency axis. Boxes
come back spanning 250 Hz to 7.75 kHz around a chirp that only occupies 3–3.5 kHz. The detection
is *there*, but the localisation — the entire point of SIAR — is gone.

So the bottleneck is a **1x1 convolution down to a small number of channels**: narrow enough
that the identity map is unrepresentable, local enough that reconstruction error stays where the
anomaly is. Plus a denoising objective (``noise_std``), which makes the identity map actively
unprofitable rather than merely difficult.
"""
from __future__ import annotations

import base64
import io
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from siar.features.patches import overlap_add, patchify
from siar.features.spec import FeatureSpec
from siar.models.registry import register_detector

__all__ = [
    "CONV_AE_FORMAT",
    "ConvAEDetector",
    "build_net",
    "latent_units",
    "max_latent_channels",
    "validate_config",
]

CONV_AE_FORMAT = "siar-conv-ae-v1"

#: Total latent units may not exceed the patch's pixel count divided by this. The exact number is
#: a judgement call; its existence is not. Without a ceiling, hyperparameter search walks
#: straight into the identity-map degeneracy described in the module docstring.
MAX_LATENT_FRACTION = 8


def latent_units(spec: FeatureSpec, config: dict) -> int:
    """Total size of the latent representation of one patch.

    The bottleneck is convolutional, so the latent is not a single number — it is
    ``latent_channels`` feature maps at the encoder's downsampled resolution. What matters for
    the identity-map question is their **product**.

    Args:
        spec: The feature spec — supplies the patch geometry.
        config: Needs ``depth`` and ``latent_channels``.

    Returns:
        ``latent_channels * (patch_frames / 2^depth) * (n_bins / 2^depth)``.
    """
    depth = int(config["depth"])
    h = spec.patch_frames // (2**depth)
    w = spec.n_bins // (2**depth)
    return int(config["latent_channels"]) * h * w


def max_latent_channels(spec: FeatureSpec, depth: int) -> int:
    """The largest ``latent_channels`` that stays inside the anti-identity budget.

    Args:
        spec: The feature spec.
        depth: Number of stride-2 stages.

    Returns:
        At least 1 — a one-channel bottleneck is always permitted, however small the patch.
    """
    h = spec.patch_frames // (2**depth)
    w = spec.n_bins // (2**depth)
    budget = (spec.patch_frames * spec.n_bins) // MAX_LATENT_FRACTION
    return max(1, budget // max(1, h * w))


def validate_config(spec: FeatureSpec, config: dict) -> None:
    """Check that an architecture is buildable and non-degenerate for this spec.

    Args:
        spec: The feature spec — supplies the patch geometry.
        config: The architecture config.

    Raises:
        ValueError: If the patch cannot be halved ``depth`` times, or the latent space is large
            enough to permit an identity map.
    """
    depth = int(config["depth"])
    channels = int(config["latent_channels"])
    factor = 2**depth

    if spec.patch_frames % factor or spec.n_bins % factor:
        raise ValueError(
            f"patch {spec.patch_frames}x{spec.n_bins} is not divisible by 2^depth={factor}; "
            f"reduce depth or change patch_frames / n_bins"
        )
    if channels < 1:
        raise ValueError(f"latent_channels must be >= 1, got {channels}")

    n_pixels = spec.patch_frames * spec.n_bins
    units = latent_units(spec, config)
    ceiling = n_pixels // MAX_LATENT_FRACTION
    if units > ceiling:
        raise ValueError(
            f"latent_channels={channels} at depth={depth} gives {units} latent units for a "
            f"{spec.patch_frames}x{spec.n_bins} patch ({n_pixels} pixels): the autoencoder "
            f"could learn the identity map and detect nothing. Maximum is {ceiling} units "
            f"({max_latent_channels(spec, depth)} channels at this depth)."
        )


def build_net(spec: FeatureSpec, config: dict):
    """Build the autoencoder.

    The bottleneck is a 1x1 convolution to ``latent_channels`` — see the module docstring. A
    ``Linear`` layer here would be narrower per unit but globally connected, and would smear an
    anomaly's reconstruction error across the entire frequency axis, which costs SIAR the only
    thing it is for.

    Args:
        spec: The feature spec — supplies the patch geometry.
        config: Keys ``depth``, ``base_channels``, ``latent_channels``, ``dropout``.

    Returns:
        A ``torch.nn.Module`` mapping ``(B, 1, patch_frames, n_bins)`` to the same shape.
    """
    import torch.nn as nn

    depth = int(config["depth"])
    base = int(config["base_channels"])
    latent_ch = int(config["latent_channels"])
    dropout = float(config.get("dropout", 0.0))

    layers: list[nn.Module] = []
    c_in = 1
    for i in range(depth):
        c_out = base * (2**i)
        layers += [
            nn.Conv2d(c_in, c_out, 3, stride=2, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        ]
        c_in = c_out
    layers.append(nn.Conv2d(c_in, latent_ch, 1))  # the bottleneck: narrow, but still local
    encoder = nn.Sequential(*layers)

    layers = [nn.Conv2d(latent_ch, c_in, 1), nn.ReLU(inplace=True)]
    for i in reversed(range(depth)):
        c_out = base * (2 ** (i - 1)) if i > 0 else base
        layers += [
            nn.ConvTranspose2d(c_in, c_out, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        ]
        c_in = c_out
    layers.append(nn.Conv2d(c_in, 1, 3, padding=1))  # linear output — the grid is unbounded
    decoder = nn.Sequential(*layers)

    class ConvAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x):
            return self.decoder(self.drop(self.encoder(x)))

    return ConvAE()


@dataclass(slots=True)
class ConvAEDetector:
    """A fitted convolutional autoencoder.

    Attributes:
        spec: The feature spec it was trained on.
        config: The architecture and training hyperparameters.
        norm_median: Median of the training patches — the centre of the input normalisation.
        norm_mad: Median absolute deviation of the training patches — its scale. Median/MAD
            rather than mean/std because the corpus contains the very outliers we are hunting,
            and they would drag a mean around.
        weights_b64: base64 of the torch ``state_dict``.
        n_params: Parameter count, for the run record.
    """

    format: ClassVar[str] = CONV_AE_FORMAT

    spec: FeatureSpec
    config: dict
    norm_median: float
    norm_mad: float
    weights_b64: str
    n_params: int = 0
    _net: Any = field(default=None, repr=False, compare=False)

    # --- normalisation ------------------------------------------------------

    def _normalise(self, patches: np.ndarray) -> np.ndarray:
        """Standardise patches with the stored training median/MAD."""
        return ((patches - self.norm_median) / (3.0 * max(self.norm_mad, 1e-6))).astype(
            np.float32
        )

    # --- training -----------------------------------------------------------

    @staticmethod
    def suggest(trial: Any, spec: FeatureSpec) -> dict:
        """Sample hyperparameters from an Optuna trial.

        The latent ceiling is derived from the patch geometry rather than hardcoded, so the
        search space automatically stays inside the non-degenerate region for whatever patch
        size was sampled.

        Args:
            trial: An ``optuna.Trial``.
            spec: The feature spec sampled for this trial.

        Returns:
            A config dict for :meth:`fit`.
        """
        factor_limit = min(int(math.log2(spec.patch_frames)), int(math.log2(spec.n_bins)))
        depth = trial.suggest_int("depth", 2, min(3, factor_limit))
        ceiling = max_latent_channels(spec, depth)
        return {
            "depth": depth,
            "base_channels": trial.suggest_categorical("base_channels", [8, 16, 32]),
            "latent_channels": trial.suggest_int("latent_channels", 1, ceiling),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "noise_std": trial.suggest_float("noise_std", 0.0, 0.3),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "epochs": trial.suggest_int("epochs", 8, 30),
        }

    @classmethod
    def default_config(cls, spec: FeatureSpec) -> dict:
        """A reasonable config for ``siar train`` when no search is being run.

        Args:
            spec: The feature spec.

        Returns:
            A config dict for :meth:`fit`.
        """
        return {
            "depth": 2,
            "base_channels": 16,
            "latent_channels": max(1, min(4, max_latent_channels(spec, 2))),
            "dropout": 0.0,
            "noise_std": 0.1,
            "lr": 1e-3,
            "weight_decay": 1e-6,
            "batch_size": 128,
            "epochs": 15,
        }

    @classmethod
    def fit(
        cls,
        grids: list[np.ndarray],
        spec: FeatureSpec,
        config: dict,
        *,
        val_grids: list[np.ndarray] | None = None,
        progress: Callable[[int, int, float, float | None], None] | None = None,
        max_patches: int = 200_000,
        seed: int = 0,
    ) -> "ConvAEDetector":
        """Learn "normal" from a corpus of grids.

        Args:
            grids: Training grids, each ``(frames, n_bins)`` ``float32``.
            spec: The feature spec the grids were built with.
            config: Hyperparameters (see :meth:`suggest`).
            val_grids: Optional held-out grids, for a per-epoch validation loss.
            progress: Optional ``(epoch, total, train_loss, val_loss)`` callback, called once per
                epoch. Optuna's pruner is driven from this.
            max_patches: Cap on training patches. Subsampled at random above this, which holds
                per-trial cost flat as the corpus grows — without it, a 10x bigger dataset makes
                every one of 100 trials 10x slower.
            seed: Seed for the subsample and for weight init.

        Returns:
            The fitted detector.

        Raises:
            ValueError: If the grids yield no usable patches, or the config is degenerate.
        """
        import torch

        validate_config(spec, config)
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        train = _stack_patches(grids, spec)
        if train.shape[0] == 0:
            raise ValueError("no usable training patches (recordings too short?)")
        if train.shape[0] > max_patches:
            train = train[rng.choice(train.shape[0], max_patches, replace=False)]

        # Normalisation stats come from the training patches only, and are frozen onto the model.
        median = float(np.median(train))
        mad = float(np.median(np.abs(train - median))) + 1e-6

        detector = cls(
            spec=spec,
            config=dict(config),
            norm_median=median,
            norm_mad=mad,
            weights_b64="",
        )

        x = torch.from_numpy(detector._normalise(train)).unsqueeze(1)  # (N, 1, T, B)
        xv = None
        if val_grids:
            val = _stack_patches(val_grids, spec)
            if val.shape[0] > max_patches // 4:
                val = val[rng.choice(val.shape[0], max_patches // 4, replace=False)]
            if val.shape[0]:
                xv = torch.from_numpy(detector._normalise(val)).unsqueeze(1)

        net = build_net(spec, config)
        n_params = sum(p.numel() for p in net.parameters())
        opt = torch.optim.Adam(
            net.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
        loss_fn = torch.nn.MSELoss()
        noise_std = float(config.get("noise_std", 0.0))
        batch = int(config["batch_size"])
        epochs = int(config["epochs"])
        n = x.shape[0]

        for epoch in range(epochs):
            net.train()
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, batch):
                xb = x[perm[i : i + batch]]
                # Denoising: corrupt the input, ask for the clean target. An identity map cannot
                # do this, so the network is pushed to learn the corpus's structure instead.
                inp = xb + torch.randn_like(xb) * noise_std if noise_std > 0 else xb
                opt.zero_grad()
                loss = loss_fn(net(inp), xb)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * xb.shape[0]
            train_loss = total / n

            val_loss = None
            if xv is not None:
                net.eval()
                with torch.no_grad():
                    vt = 0.0
                    for i in range(0, xv.shape[0], batch):
                        vb = xv[i : i + batch]
                        vt += float(loss_fn(net(vb), vb)) * vb.shape[0]
                    val_loss = vt / xv.shape[0]

            if progress:
                progress(epoch + 1, epochs, train_loss, val_loss)

        net.eval()
        detector.weights_b64 = _weights_to_b64(net.state_dict())
        detector.n_params = int(n_params)
        detector._net = net
        return detector

    # --- scoring ------------------------------------------------------------

    def _net_eval(self):
        """Return the loaded network, restoring it from base64 on first use."""
        if self._net is None:
            import torch

            net = build_net(self.spec, self.config)
            buf = io.BytesIO(base64.b64decode(self.weights_b64))
            # weights_only=True: a model file can never execute code on load.
            net.load_state_dict(torch.load(buf, map_location="cpu", weights_only=True))
            net.eval()
            self._net = net
        return self._net

    def error_map(self, grid: np.ndarray, *, batch: int = 256) -> np.ndarray:
        """Score one grid, per pixel.

        Args:
            grid: ``(frames, n_bins)`` ``float32``, built with :attr:`spec`.
            batch: Patches per forward pass.

        Returns:
            A ``(frames, n_bins)`` ``float32`` error map, trimmed back to the input's frame count
            if the grid was edge-padded to fill a patch. Raw squared error — normalisation and
            thresholding belong to :mod:`siar.detect`.
        """
        import torch

        g = np.asarray(grid, dtype=np.float32)
        if g.shape[0] == 0:
            return np.zeros((0, self.spec.n_bins), dtype=np.float32)

        patches, starts, padded_frames = patchify(
            g, self.spec.patch_frames, self.spec.stride_frames
        )
        x = torch.from_numpy(self._normalise(patches)).unsqueeze(1)

        net = self._net_eval()
        errors = np.empty(
            (x.shape[0], self.spec.patch_frames, self.spec.n_bins), dtype=np.float32
        )
        with torch.no_grad():
            for i in range(0, x.shape[0], batch):
                xb = x[i : i + batch]
                # The residual is NOT reduced. Keeping all (T, B) of it is what lets box
                # extraction localise an anomaly in frequency as well as time.
                err = (net(xb) - xb) ** 2
                errors[i : i + xb.shape[0]] = err.squeeze(1).numpy()

        emap = overlap_add(errors, starts, padded_frames)
        return emap[: g.shape[0]]

    # --- serialisation ------------------------------------------------------

    def to_json(self) -> dict:
        """Serialise to a JSON-safe descriptor, weights included."""
        return {
            "format": CONV_AE_FORMAT,
            "detector": "conv_ae",
            "spec": self.spec.to_dict(),
            "config": dict(self.config),
            "norm_median": float(self.norm_median),
            "norm_mad": float(self.norm_mad),
            "n_params": int(self.n_params),
            "weights_b64": self.weights_b64,
        }

    @classmethod
    def from_json(cls, obj: dict) -> "ConvAEDetector":
        """Rebuild a detector from :meth:`to_json` output.

        Args:
            obj: The descriptor.

        Returns:
            The restored detector.

        Raises:
            ValueError: If the descriptor is malformed or of the wrong format.
        """
        fmt = obj.get("format")
        if fmt != CONV_AE_FORMAT:
            raise ValueError(f"expected format {CONV_AE_FORMAT!r}, got {fmt!r}")
        if not obj.get("weights_b64"):
            raise ValueError("model descriptor has no weights")
        return cls(
            spec=FeatureSpec.from_dict(obj["spec"]),
            config=dict(obj["config"]),
            norm_median=float(obj["norm_median"]),
            norm_mad=float(obj["norm_mad"]),
            weights_b64=str(obj["weights_b64"]),
            n_params=int(obj.get("n_params", 0)),
        )


# Registered after the class body so the decorator sees a complete class.
register_detector("conv_ae")(ConvAEDetector)


def _stack_patches(grids: list[np.ndarray], spec: FeatureSpec) -> np.ndarray:
    """Patchify every grid and stack the results.

    Args:
        grids: Grids, each ``(frames, n_bins)``.
        spec: Supplies the patch geometry.

    Returns:
        ``(total_patches, patch_frames, n_bins)`` ``float32``. Empty if no grid had any frames.
    """
    out: list[np.ndarray] = []
    for g in grids:
        if g.shape[0] == 0:
            continue
        patches, _starts, _frames = patchify(g, spec.patch_frames, spec.stride_frames)
        out.append(patches)
    if not out:
        return np.zeros((0, spec.patch_frames, spec.n_bins), dtype=np.float32)
    return np.concatenate(out, axis=0)


def _weights_to_b64(state_dict) -> str:
    """Serialise a torch ``state_dict`` to base64."""
    import torch

    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")
