#!/usr/bin/env bash
# Sets up everything needed to run the rover's software stack:
#   1. Python venv with all dependencies (NLP voice app + RL simulation)
#   2. The offline Vosk speech model
#
# The Arduino firmware (firmware/Rover_Main.ino) is flashed separately
# via the Arduino IDE -- see the README for wiring + library requirements.
#
# Usage:
#   bash setup.sh

set -e

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

echo "Creating virtual environment in $VENV_DIR ..."
$PYTHON_BIN -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing voice-control app (rover_nlp_app) dependencies..."
pip install -r rover_nlp_app/requirements.txt

echo "Installing RL simulation (rl_rover_v2) dependencies..."
pip install -r rl_rover_v2/rl_rover_v2/requirements.txt

echo "Fetching offline Vosk speech model..."
bash rover_nlp_app/setup.sh

echo ""
echo "Setup complete. Activate the environment with:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "Then run:"
echo "    streamlit run rover_nlp_app/app.py        # voice control dashboard"
echo "    python rl_rover_v2/rl_rover_v2/run_all.py  # RL navigation simulation"
