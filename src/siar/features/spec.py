# Vixen Intelligence c.2026
"""``FeatureSpec`` — the contract between audio and the model.

A :class:`FeatureSpec` completely determines the picture a detector sees. It is computed at
training time, **stored on the trained model**, and re-applied verbatim at inference. That is
what makes a model re-runnable on a new folder: the model does not carry "a spectrogram", it
carries the exact recipe for building the one it was trained on.

Get this wrong and the failure is silent and total — a model trained on a 48 kHz, 96-bin grid
scored against a 44.1 kHz, 64-bin grid does not error, it just draws boxes in the wrong places.
So the spec is frozen, hashable, and validated at construction.

This module imports nothing heavier than numpy: ``siar scan`` and the CLI can read a spec
without paying for torch.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

__all__ = ["FEATURE_MODES", "FeatureSpec"]

#: The two ways SIAR can lay out the frequency axis.
#:
#: ``pooled_linear`` — crop the linear FFT bins to the band and mean-pool them into ``n_bins``
#:   equal-width groups. Frequency resolution is uniform in Hz.
#: ``log_mel`` — a mel filterbank over the band. Resolution is fine at low frequency and coarse
#:   at high, which matches how most biological and mechanical sound is structured.
#:
#: Which one wins is data-dependent, so it is a searched hyperparameter, not a fixed choice.
FEATURE_MODES = ("pooled_linear", "log_mel")


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """The full recipe for turning a waveform into the grid a detector sees.

    Attributes:
        mode: One of :data:`FEATURE_MODES`.
        sample_rate: Target sample rate in Hz. Audio at any other rate is resampled to this,
            so every grid in a corpus is directly comparable.
        fft_size: FFT size in samples. Even; a power of two in practice.
        hop_size: Hop between frames in samples. Sets the time resolution, ``hop/sr`` seconds.
        window: Analysis window name (``"hann"`` in practice).
        fmin_hz: Low edge of the analysed band.
        fmax_hz: High edge of the analysed band. Must not exceed Nyquist.
        n_bins: Height of the grid — pooled groups, or mel filters. The net's input width.
        patch_frames: Width, in frames, of the patches the detector reconstructs.
        stride_frames: Hop, in frames, between successive patches.

    Raises:
        ValueError: If any field is out of range or the fields are mutually inconsistent.
    """

    mode: str
    sample_rate: int
    fft_size: int
    hop_size: int
    window: str
    fmin_hz: float
    fmax_hz: float
    n_bins: int
    patch_frames: int
    stride_frames: int

    def __post_init__(self) -> None:
        if self.mode not in FEATURE_MODES:
            raise ValueError(f"mode must be one of {FEATURE_MODES}, got {self.mode!r}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.fft_size <= 0 or self.fft_size % 2 != 0:
            raise ValueError(f"fft_size must be positive and even, got {self.fft_size}")
        if self.hop_size <= 0:
            raise ValueError(f"hop_size must be positive, got {self.hop_size}")
        nyquist = self.sample_rate / 2.0
        if not 0.0 <= self.fmin_hz < self.fmax_hz:
            raise ValueError(f"need 0 <= fmin < fmax, got {self.fmin_hz} .. {self.fmax_hz}")
        if self.fmax_hz > nyquist:
            raise ValueError(
                f"fmax_hz ({self.fmax_hz}) exceeds Nyquist ({nyquist}) for "
                f"sample_rate {self.sample_rate}"
            )
        # The autoencoder halves the grid twice, so both patch axes must survive two exact
        # divisions by 2. Enforced here rather than in the model, because it is a property of
        # the *picture*, and a spec that no detector can consume is not a valid spec.
        if self.n_bins <= 0 or self.n_bins % 4 != 0:
            raise ValueError(f"n_bins must be a positive multiple of 4, got {self.n_bins}")
        if self.patch_frames <= 0 or self.patch_frames % 4 != 0:
            raise ValueError(
                f"patch_frames must be a positive multiple of 4, got {self.patch_frames}"
            )
        if not 0 < self.stride_frames <= self.patch_frames:
            raise ValueError(
                f"need 0 < stride_frames <= patch_frames, got {self.stride_frames} "
                f"vs {self.patch_frames}"
            )

    @property
    def delta_t(self) -> float:
        """Seconds between successive frames (``hop_size / sample_rate``)."""
        return self.hop_size / self.sample_rate

    def to_dict(self) -> dict:
        """Return the spec as a plain JSON-safe dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: dict) -> "FeatureSpec":
        """Rebuild a spec from :meth:`to_dict` output.

        Args:
            obj: A dict carrying every field of the spec.

        Returns:
            The validated :class:`FeatureSpec`.

        Raises:
            ValueError: If a field is missing or invalid.
        """
        fields = cls.__dataclass_fields__
        missing = set(fields) - set(obj)
        if missing:
            raise ValueError(f"FeatureSpec is missing field(s): {sorted(missing)}")
        return cls(**{k: obj[k] for k in fields})

    def cache_key(self) -> str:
        """A stable short hash of every field.

        Used to name the feature-cache directory, so two specs that differ in any way — even
        one that does not change the grid's shape — never share cached features.

        Returns:
            A 16-character hex digest.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
