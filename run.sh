#!/usr/bin/env bash
# One command to set up and launch Chatbot Lab.
# Usage:  ./run.sh
set -e
cd "$(dirname "$0")"

# 1. Create an isolated Python environment (first run only).
if [ ! -d ".venv" ]; then
  echo "==> Creating virtual environment (.venv)"
  python3.11 -m venv .venv
fi
source .venv/bin/activate

# 2. Install dependencies.
echo "==> Installing dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# 3. Make sure a config exists.
if [ ! -f "config.yaml" ]; then
  echo "==> Creating config.yaml from the example (edit it, then re-run)"
  cp config.example.yaml config.yaml
fi

# 4. Launch.
echo "==> Starting Chatbot Lab at http://127.0.0.1:8000"
python app.py
