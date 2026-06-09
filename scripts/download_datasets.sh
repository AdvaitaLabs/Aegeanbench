#!/usr/bin/env bash
#
# Download historical football datasets from Kaggle into data/kaggle/.
#
# Prerequisite — Kaggle API credentials (one-time setup):
#   1. Go to https://www.kaggle.com/settings -> "Create New API Token"
#   2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
#   3. chmod 600 ~/.kaggle/kaggle.json
#
# Then run:
#   bash scripts/download_datasets.sh
#
# The script is idempotent: existing files are skipped unless --force is given.

set -euo pipefail

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

# Resolve repo root regardless of where the script is invoked from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/data/kaggle"

mkdir -p "${TARGET_DIR}"

# ----------------------------------------------------------------------
# kaggle CLI bootstrap
# ----------------------------------------------------------------------
if ! command -v kaggle >/dev/null 2>&1; then
    echo "[+] Installing kaggle CLI via pip (user scope)…"
    pip install --user --quiet kaggle
    # Make sure the user pip bin dir is on PATH
    export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
    cat >&2 <<EOF
[!] Kaggle credentials not found at \$HOME/.kaggle/kaggle.json
    1. Visit https://www.kaggle.com/settings and click "Create New API Token"
    2. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json
    3. chmod 600 ~/.kaggle/kaggle.json
    Then re-run this script.
EOF
    exit 1
fi

chmod 600 "$HOME/.kaggle/kaggle.json" || true

# ----------------------------------------------------------------------
# dataset list — slug : subdir
# ----------------------------------------------------------------------
declare -a DATASETS=(
    # International results since 1872 — small, very useful for Elo / form
    "martj42/international-football-results-from-1872-to-2017:international_results"
    # FIFA World Cup all matches
    "abecklas/fifa-world-cup:fifa_world_cup"
    # Player ratings (FIFA video game database, used as proxy for skill)
    "stefanoleone992/fifa-22-complete-player-dataset:fifa_player_ratings"
)

download_one() {
    local slug="$1"
    local subdir="$2"
    local dest="${TARGET_DIR}/${subdir}"

    if [[ -d "$dest" && "$(ls -A "$dest" 2>/dev/null)" && "$FORCE" -eq 0 ]]; then
        echo "[=] ${subdir} already present, skipping (use --force to refresh)"
        return 0
    fi

    echo "[+] Downloading ${slug} -> ${dest}"
    mkdir -p "$dest"
    kaggle datasets download -d "$slug" -p "$dest" --unzip --quiet
    echo "[✓] ${subdir}: $(ls "$dest" | wc -l) files"
}

for entry in "${DATASETS[@]}"; do
    slug="${entry%%:*}"
    subdir="${entry##*:}"
    download_one "$slug" "$subdir" || echo "[!] failed: $slug (continuing)"
done

echo
echo "[done] Datasets in: ${TARGET_DIR}"
du -sh "${TARGET_DIR}"/*/ 2>/dev/null || true
