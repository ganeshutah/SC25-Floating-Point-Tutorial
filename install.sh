#!/bin/bash
set -e
echo "Installing..."
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install transformers bitsandbytes accelerate
if command -v nixnan &>/dev/null; then echo "✓ nixnan found"; else echo "⚠ nixnan not found"; fi
echo "✓ Complete"
