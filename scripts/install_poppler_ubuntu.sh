#!/usr/bin/env bash
set -euo pipefail

if command -v pdftoppm >/dev/null 2>&1; then
    echo "Poppler utilities already installed; skipping."
    exit 0
fi

sudo apt-get update
sudo apt-get install -y poppler-utils

echo "Poppler utilities installed."
