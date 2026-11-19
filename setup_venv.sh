#!/bin/bash

set -euo pipefail

python_version="3.14"
venv_name="venv"
force_recreate=false

if [ "${1:-}" = "-f" ]; then
    force_recreate=true
    export UV_VENV_CLEAR=1
fi

if [ -d "$venv_name" ] && [ "$force_recreate" = false ]; then
    echo "Virtual environment '$venv_name' already exists."
    echo "Activate using '. $venv_name/bin/activate'"
    exit 0
fi

echo "Creating virtual environment '$venv_name'..."

uv venv "$venv_name" --python "$python_version"
. "$venv_name/bin/activate"

echo "Installing required packages..."
uv pip install -r requirements.txt

echo "✓ Done! Activate using '. $venv_name/bin/activate'"
