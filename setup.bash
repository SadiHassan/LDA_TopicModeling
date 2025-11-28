#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Downloading spaCy English model..."
python -m spacy download en_core_web_md

echo "Setup complete!"
