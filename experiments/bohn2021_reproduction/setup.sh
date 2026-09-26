#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
BOHN_ENV="${BOHN_ENV:-/home/mapples/.local/share/bohn2021-python37}"
BOHN_ART="$(pwd)/research_artifacts/bohn2021_reproduction_2026-09-17"
mkdir -p "$BOHN_ART/sources"
if [ ! -x "$BOHN_ENV/bin/python" ]; then
  curl --fail -L https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh -o "$BOHN_ART/sources/miniconda37.sh"
  bash "$BOHN_ART/sources/miniconda37.sh" -b -p "$BOHN_ENV"
fi
"$BOHN_ENV/bin/pip" install -r experiments/bohn2021_reproduction/requirements-legacy.txt
while read -r repo folder commit; do
  if [ ! -d "$BOHN_ART/sources/$folder" ]; then
    git clone "https://github.com/eivindeb/$repo.git" "$BOHN_ART/sources/$folder"
    git -C "$BOHN_ART/sources/$folder" checkout --detach "$commit"
  fi
  actual=$(git -C "$BOHN_ART/sources/$folder" rev-parse HEAD)
  [ "$actual" = "$commit" ] || { echo "Unexpected revision in $folder: $actual"; exit 1; }
done <<'COMMITS'
gym-letMPC gym-horizon 3f0572e4761f6797327e48d77e2abc343ab0239c
stable-baselines stable-baselines-horizon 1539282c8a11417b96e13040e5c4649be251dca2
do-mpc do-mpc-horizon 3f4fea1d262082083ac280b5206a7ca915a6676d
COMMITS
"$BOHN_ENV/bin/python" experiments/bohn2021_reproduction/configure.py
"$BOHN_ENV/bin/python" experiments/bohn2021_reproduction/validate.py
