# Tutorial 04 — Build for deployment

How to turn the source repository into a distributable wheel with compiled, obfuscated Python,
and set up the API-key gate so the binary only runs for people you have issued a key to.

**Time:** two minutes, once the build dependencies are in place.

---

## What you get

A normal `pip install -e .` uses `.py` source directly — that is development mode. The
deployment build compiles every module to a native `.so` extension via Cython, strips the `.py`
and `.c` source from the wheel, and leaves only:

- **compiled `.so` files** — one per module, not human-readable
- **`__init__.py` stubs** — thin package markers (imports and `__all__` only)
- **`web/static/`** — the dashboard's HTML, JS and CSS

The result is a standard Python wheel that installs with `pip` on any machine with the same
Python version and platform. No Cython needed on the target — it is a build-time dependency
only.

---

## Prerequisites

On the **build machine** (your dev box), you need:

```bash
pip install 'Cython>=3.0'
```

Plus the Python development headers for your Python version. On Ubuntu/Debian:

```bash
sudo apt install python3.13-dev      # match your Python version
```

On macOS (Xcode command-line tools provide them), or in a conda environment, the headers are
usually already present.

---

## Step 1 — build the wheel

```bash
pip wheel --no-build-isolation -w dist/ --no-deps .
```

That runs Cython on every `.py` module, compiles the generated C to native extensions, and
packages the result as a `.whl` file in `dist/`.

```
dist/siar-0.1.0-cp313-cp313-linux_x86_64.whl
```

The wheel is platform-specific (`linux_x86_64`, `macosx_arm64`, etc.) because it contains
compiled C. You build on the same OS and architecture as the target.

### Verify the wheel is clean

```bash
unzip -l dist/siar-*.whl | grep -E '\.(py|c|so)$'
```

You should see:

- `*.cpython-313-x86_64-linux-gnu.so` — every compiled module
- `*/__init__.py` — package stubs only
- **no** `*.c` files, **no** non-init `*.py` files

---

## Step 2 — set up the API-key gate

Before shipping, activate the key on the target machine. This is a one-time step — it stores a
salted SHA-256 hash in the SIAR workspace.

```bash
siar activate <your-secret-key>
```

```
activated — key hash written to /home/user/.siar/.api_key
export SIAR_API_KEY=<your-key> in your shell to use SIAR
```

Every command except `siar version` and `siar activate` now requires `$SIAR_API_KEY` to be set
in the environment:

```bash
export SIAR_API_KEY=<your-secret-key>
siar models                               # works
siar train /data/audio --name baseline     # works
```

Without it:

```bash
unset SIAR_API_KEY
siar models
# siar: $SIAR_API_KEY is not set — export it in your shell to use SIAR
```

Wrong key:

```bash
SIAR_API_KEY=wrong siar models
# siar: invalid API key
```

### Issuing keys

You choose the key — it is any string. Issue a different key per client and activate it on their
machine before shipping.  The key itself is never stored; only the salted hash lives on disk at
`~/.siar/.api_key` (mode 600).

---

## Step 3 — install on the target machine

Copy the wheel and install it with pip:

```bash
pip install siar-0.1.0-cp313-cp313-linux_x86_64.whl
```

That is it. The `siar` console script is available immediately:

```bash
export SIAR_API_KEY=<the-key-you-issued>
siar version
siar train /data/recordings --name site-a
siar run <model-uid> /data/new-recordings
siar dash --open
```

No source code is on the machine. The installed package contains only compiled extensions.

---

## Deployment checklist

```
[ ] Build machine has Cython >= 3.0 and Python dev headers
[ ] pip wheel --no-build-isolation -w dist/ --no-deps .
[ ] Verify: no .py source (except __init__.py) or .c files in the wheel
[ ] Copy wheel to target machine
[ ] pip install siar-*.whl on target
[ ] siar activate <key> on target
[ ] Client sets export SIAR_API_KEY=<key> in their shell profile
[ ] siar version — confirms install
[ ] siar train / run / dash — confirms the gate passes
```

---

## What the gate does and does not protect

The API key is a **barrier**, not a lock. It stops someone from running `siar train` on a
machine where you have not issued a key. It does not stop a determined reverse-engineer from
disassembling the `.so` files — that is always possible with compiled C extensions, just
materially harder than reading `.py` source.

The two layers together — compiled extensions (no readable source) plus the key gate (no casual
execution) — are the practical ceiling for a pip-installable Python tool.

---

## Development workflow (unchanged)

None of this affects your own development cycle. An editable install bypasses Cython entirely:

```bash
pip install -e .                          # uses .py source directly
pytest                                    # runs against source
```

The Cython build only kicks in when you build a wheel for distribution.
