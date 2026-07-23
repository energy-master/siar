#!/usr/bin/env bash
# Vixen Intelligence c.2026
#
# Rebuild the compiled/obfuscated SIAR wheel and copy it into the siar-dist
# shipping repo, in one step.
#
#   .py  --(Cython)-->  .c  --(cc)-->  .so  -->  wheel  -->  siar-dist/dist/
#
# Usage:
#   ./build-dist.sh              build the wheel and copy it into siar-dist/dist/
#   ./build-dist.sh --release    ...then publish it as a GitHub release asset
#
# --release needs a GitHub token with 'contents: write' on the dist repo, in
# $GITHUB_TOKEN or $GH_TOKEN.  The release is tagged vX.Y.Z from the wheel
# version and the wheel is attached as a downloadable asset, installable with:
#   pip install https://github.com/<owner>/siar-dist/releases/download/vX.Y.Z/<wheel>
#
# Development installs (pip install -e .) are unaffected — this only runs the
# distribution build.  See tutorials/04-build-for-deployment.md for the details.

set -euo pipefail

SRC_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_REPO="${SIAR_DIST_REPO:-$(cd "$SRC_REPO/../siar-dist" && pwd)}"

# ---- arguments -------------------------------------------------------------
DO_RELEASE=0
for arg in "$@"; do
    case "$arg" in
        --release) DO_RELEASE=1 ;;
        -h|--help)
            sed -n '3,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "error: unknown argument '$arg' (see --help)" >&2; exit 1 ;;
    esac
done

echo "==> source repo:  $SRC_REPO"
echo "==> dist repo:    $DIST_REPO"

if [[ ! -d "$DIST_REPO/dist" ]]; then
    echo "error: '$DIST_REPO/dist' not found — is siar-dist checked out next to siar?" >&2
    exit 1
fi

# Fail early on --release if the token is missing, before spending a build on it.
if [[ "$DO_RELEASE" == 1 ]]; then
    TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
    if [[ -z "$TOKEN" ]]; then
        echo "error: --release needs a GitHub token in \$GITHUB_TOKEN or \$GH_TOKEN" >&2
        exit 1
    fi
fi

# ---- 1. clean --------------------------------------------------------------
echo "==> cleaning previous build artefacts"
rm -rf "$SRC_REPO/build" "$SRC_REPO/dist"
find "$SRC_REPO/src/siar" -name '*.c' -delete          # Cython-generated, safe to drop

# ---- 2. build --------------------------------------------------------------
echo "==> building wheel"
pip wheel --no-build-isolation -w "$SRC_REPO/dist" --no-deps "$SRC_REPO"

WHEEL="$(ls -t "$SRC_REPO"/dist/siar-*.whl 2>/dev/null | head -n1)"
if [[ -z "${WHEEL:-}" ]]; then
    echo "error: no wheel produced in $SRC_REPO/dist" >&2
    exit 1
fi
echo "==> built: $(basename "$WHEEL")"

# ---- 3. verify no readable source leaked -----------------------------------
echo "==> verifying wheel contains no readable source"
LEAKED="$(unzip -Z1 "$WHEEL" | grep -E '\.(py|c)$' | grep -v '/__init__\.py$' || true)"
if [[ -n "$LEAKED" ]]; then
    echo "error: wheel contains readable source files:" >&2
    echo "$LEAKED" >&2
    exit 1
fi

# ---- 4. copy into the shipping repo ----------------------------------------
echo "==> copying wheel into $DIST_REPO/dist"
rm -f "$DIST_REPO"/dist/siar-*.whl
cp "$WHEEL" "$DIST_REPO/dist/"
echo "==> shipped: $DIST_REPO/dist/$(basename "$WHEEL")"

# ---- 5. optional GitHub release --------------------------------------------
if [[ "$DO_RELEASE" == 1 ]]; then
    base="$(basename "$WHEEL")"
    ver="$(printf '%s' "$base" | sed -E 's/^siar-([^-]+)-.*/\1/')"   # siar-0.1.0-... -> 0.1.0
    tag="v$ver"

    origin="$(git -C "$DIST_REPO" remote get-url origin)"
    slug="$(printf '%s' "$origin" | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')"
    api="https://api.github.com/repos/$slug"
    auth=(-H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json")

    echo "==> publishing release $tag to $slug"

    # Find an existing release for this tag, or create one.
    rel_id="$(curl -fsS "${auth[@]}" "$api/releases/tags/$tag" 2>/dev/null \
        | python3 -c 'import sys,json; print(json.load(sys.stdin).get("id",""))' 2>/dev/null)" || rel_id=""

    if [[ -z "$rel_id" ]]; then
        echo "==> creating release $tag"
        rel_id="$(curl -fsS "${auth[@]}" -X POST "$api/releases" \
            -d "{\"tag_name\":\"$tag\",\"name\":\"SIAR $ver\",\"body\":\"SIAR $ver — compiled wheel (cp313, linux_x86_64).\"}" \
            | python3 -c 'import sys,json; print(json.load(sys.stdin).get("id",""))')" || rel_id=""
        if [[ -z "$rel_id" ]]; then
            echo "error: failed to create release $tag" >&2; exit 1
        fi
    else
        echo "==> reusing existing release $tag (id $rel_id)"
        # Remove a same-named asset so the upload doesn't collide (HTTP 422).
        asset_id="$(curl -fsS "${auth[@]}" "$api/releases/$rel_id/assets" \
            | python3 -c "import sys,json; print(next((a['id'] for a in json.load(sys.stdin) if a['name']=='$base'),''))")" || asset_id=""
        if [[ -n "$asset_id" ]]; then
            echo "==> replacing existing asset $base"
            curl -fsS "${auth[@]}" -X DELETE "$api/releases/assets/$asset_id" >/dev/null
        fi
    fi

    echo "==> uploading $base"
    status="$(curl -s -o /dev/null -w '%{http_code}' "${auth[@]}" \
        -H "Content-Type: application/octet-stream" \
        --data-binary @"$WHEEL" \
        "https://uploads.github.com/repos/$slug/releases/$rel_id/assets?name=$base")"
    if [[ "$status" != "201" ]]; then
        echo "error: asset upload failed (HTTP $status)" >&2; exit 1
    fi

    echo "==> released:"
    echo "    https://github.com/$slug/releases/download/$tag/$base"
    echo "    pip install https://github.com/$slug/releases/download/$tag/$base"
fi

echo "==> done"
