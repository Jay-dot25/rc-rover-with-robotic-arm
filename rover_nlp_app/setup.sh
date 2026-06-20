#!/usr/bin/env bash
# Downloads the offline Vosk speech model into the expected vosk-model/ folder.
# The 68MB model is NOT committed to git -- this script fetches it fresh.
set -e

cd "$(dirname "$0")"

MODEL_ZIP="vosk-model-small-en-us-0.15.zip"
MODEL_URL="https://alphacephei.com/vosk/models/${MODEL_ZIP}"

if [ -d "vosk-model/am" ]; then
    echo "Vosk model already present, skipping download."
    exit 0
fi

echo "Downloading Vosk offline speech model (~40MB)..."
curl -L -o "$MODEL_ZIP" "$MODEL_URL"

echo "Extracting..."
unzip -q "$MODEL_ZIP"
mkdir -p vosk-model
mv vosk-model-small-en-us-0.15/* vosk-model/
rmdir vosk-model-small-en-us-0.15
rm "$MODEL_ZIP"

echo "Vosk model ready at rover_nlp_app/vosk-model/"
