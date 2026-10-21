#!/usr/bin/env bash
# Stage 5 vectors on a laptop: makes a private Python environment in ~/.cache/folio/venv
# the first time (nothing else on the machine changes), installs fastembed there and runs
# embed_search.py on the folder this script sits in. Run it again after new questions
# arrive: only the new texts are embedded.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
venv="$HOME/.cache/folio/venv"
if ! "$venv/bin/python" -c "import fastembed" >/dev/null 2>&1; then
  rm -rf "$venv"
  mkdir -p "$(dirname "$venv")"
  for py in /usr/bin/python3 python3 "$HOME/anaconda3/bin/python3"; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -m venv "$venv" >/dev/null 2>&1; then
      break
    fi
    rm -rf "$venv"
  done
  if [ ! -x "$venv/bin/python" ]; then
    echo "Could not create a Python environment. On Ubuntu: sudo apt install python3-venv" >&2
    exit 1
  fi
  echo "Installing fastembed in $venv (one time only)..."
  "$venv/bin/python" -m pip install --quiet --upgrade pip
  "$venv/bin/python" -m pip install --quiet "fastembed==0.9.0"
fi
exec "$venv/bin/python" "$here/embed_search.py" "$here"
