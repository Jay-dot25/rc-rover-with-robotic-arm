# RC Rover with Robotic Arm
### Bluetooth-Controlled Rover · 6-DOF Arm · Voice Control (NLP) · Reinforcement-Learning Autonomy

A Bluetooth-controlled robotic rover with a 6-DOF robotic arm, ultrasonic obstacle detection with auto-reverse safety override, and a custom Android app for real-time control. Extended with two AI subsystems: an offline NLP voice-command pipeline and a Double Dueling DQN reinforcement-learning agent that learns autonomous navigation in simulation.

---

## System Overview

```
┌──────────────────────────┐        Bluetooth (HC-05)         ┌────────────────────────────┐
│   Android App (Kotlin)    │ ────────────────────────────────▶│   Arduino Mega Firmware     │
│   app/rovercontroller.apk │  single-char commands: F B L R S  │   firmware/Rover_Main.ino   │
└──────────────────────────┘  + slider protocol: A###A...       └──────────────┬─────────────┘
                                                                                 │
┌──────────────────────────┐        Bluetooth (HC-05)                          │ PWM (PCA9685)
│  Offline Voice Control     │ ─────────────────────────────────────────────────┤
│  (Streamlit + Vosk +       │  same char protocol as the app                   ▼
│   Conv1D CNN NLP)           │                                       ┌──────────────────────┐
│  rover_nlp_app/app.py       │                                       │  L298N motor driver   │
└──────────────────────────┘                                       │  + 6-DOF servo arm    │
                                                                       │  + HC-SR04 ultrasonic │
                                                                       └──────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│   RL Autonomy Simulation (Pygame, no hardware needed)            │
│   Double Dueling DQN learns to navigate to a goal                │
│   while avoiding obstacles — purely in simulation                 │
│   rl_rover_v2/rl_rover_v2/                                       │
└────────────────────────────────────────────────────────────────┘
```

## Hardware

| Component | Role |
|-----------|------|
| Arduino Mega 2560 | Main controller |
| HC-05 | Bluetooth serial link to phone / PC |
| L298N | Dual H-bridge motor driver (drive motors) |
| PCA9685 (Adafruit_PWMServoDriver) | 16-channel PWM driver for the 6-DOF arm servos |
| HC-SR04 | Ultrasonic distance sensor (obstacle detection) |
| MIT App Inventor Android app | Manual drive + arm control over Bluetooth |

**Wiring reference:** see `pictures/hardware_connections.jpg` and `pictures/Full_Body.jpg`.

## Communication Protocol

The firmware listens on Bluetooth Serial (`Serial1`) for a simple character protocol:

| Input | Meaning |
|-------|---------|
| `F` / `B` / `L` / `R` / `S` | Forward / Backward / Left / Right / Stop |
| `A###A` | Set servo channel **A** (gripper base) to angle `###` (0-180), e.g. `A090A` |
| `C###C` | Set servo **C** to angle `###` (servo D auto-mirrors as `180 - value`) |
| `D###D`, `E###E`, `G###G`, `H###H` | Set the corresponding arm joint to angle `###` |

The same protocol is used by both the Android app and the offline voice-control app, so either can drive the rover interchangeably.

**Safety override:** if the rover is moving forward and the ultrasonic sensor detects an obstacle within `OBSTACLE_DISTANCE` (default 30cm), the firmware automatically reverses for 300ms and stops, regardless of what command stream is coming in.

## Subsystems

### 1. Firmware (`firmware/Rover_Main.ino`)
Arduino sketch handling motor control, the 6-DOF arm via PCA9685, ultrasonic-based safety override, and the Bluetooth command parser described above.

**Required Arduino libraries:** `Adafruit_PWMServoDriver` (install via Library Manager).

Flash with the Arduino IDE, board = **Arduino Mega 2560**.

### 2. Android App (`app/`)
Pre-built APK/AAB for manual driving and arm control. Screens in `pictures/app_ui_1.jpg` and `pictures/app_ui_2.jpg`. Install `app/rovercontroller.apk` directly on an Android device, or `app/RoverControl.aab` for Play Store distribution.

### 3. Offline Voice Control -- "CHIP" (`rover_nlp_app/`)
A Streamlit dashboard that listens for a wake word ("Hey CHIP"), transcribes speech fully offline with **Vosk**, classifies the resulting text into a drive command using a small **Conv1D CNN** trained in-session, and forwards the command to the rover over the same Bluetooth protocol via `pyserial`.

```bash
cd rover_nlp_app
bash setup.sh              # downloads the offline Vosk model
streamlit run app.py
```

### 4. RL Autonomy Simulation (`rl_rover_v2/`)
A from-scratch **Double Dueling DQN** agent (NumPy, no deep-learning framework) trained in a Pygame physics simulation to navigate to a goal while avoiding obstacles, with curriculum learning (easy to medium to hard) and prioritized experience replay. This subsystem is simulation-only -- it doesn't require the physical rover or any hardware.

```bash
cd rl_rover_v2/rl_rover_v2
pip install -r requirements.txt
python run_all.py
```

See `rl_rover_v2/rl_rover_v2/README.md` for the full architecture writeup (Double DQN, Dueling heads, PER, reward shaping, curriculum stages) and training results (96% goal-reach rate over the last 200 of 2000 episodes).

## Setup (First Time)

```bash
git clone <your-repo-url>
cd rc-rover-with-robotic-arm

# Sets up Python venv + deps for the NLP app and RL simulation,
# and downloads the offline Vosk speech model
bash setup.sh
```

Flashing the firmware is separate -- open `firmware/Rover_Main.ino` in the Arduino IDE, install the `Adafruit_PWMServoDriver` library, select **Arduino Mega 2560**, and upload.

## Running

```bash
source .venv/bin/activate

# Voice-controlled dashboard (needs the physical rover + HC-05 over Bluetooth)
streamlit run rover_nlp_app/app.py

# RL navigation simulation (no hardware needed)
python rl_rover_v2/rl_rover_v2/run_all.py
```

## Project Structure

```
rc-rover-with-robotic-arm/
├── firmware/
│   └── Rover_Main.ino          # Arduino Mega sketch: motors, arm, ultrasonic, BT parser
├── app/
│   ├── rovercontroller.apk     # Android control app (install directly)
│   └── RoverControl.aab        # Android App Bundle (Play Store format)
├── rover_nlp_app/
│   ├── app.py                  # Streamlit offline voice-control dashboard ("CHIP")
│   ├── requirements.txt
│   └── setup.sh                # Downloads the Vosk offline speech model
├── rl_rover_v2/rl_rover_v2/
│   ├── run_all.py              # Single entry point: train + simulate
│   ├── src/
│   │   ├── env.py              # RoverEnv: 8 sensors, 5 actions, 12-D state
│   │   ├── agent.py            # Double Dueling DQN + Prioritized Experience Replay
│   │   ├── train.py            # Training loop with curriculum learning
│   │   ├── simulate.py         # Live Pygame window + video recorder
│   │   └── plot_results.py     # Training dashboard + charts
│   ├── outputs/                # Saved checkpoints, training logs, plots
│   ├── requirements.txt
│   └── README.md               # Full RL architecture writeup
├── pictures/
│   ├── Full_Body.jpg
│   ├── hardware_connections.jpg
│   ├── app_ui_1.jpg
│   └── app_ui_2.jpg
├── setup.sh                    # Top-level setup: venv + deps + Vosk model
└── LICENSE
```

## Known Issues / Lessons Learned

**HC-05 brownout during arm movement (fixed):** the HC-05 module would lose power and disconnect whenever a servo moved under load, because the L298N's onboard 5V regulator couldn't supply both the drive motors and the servo bus simultaneously. Fixed by routing a dedicated buck converter's 5V output directly to the Arduino's 5V rail instead of relying on the L298N regulator, isolating logic-level power from motor power.

**`firmware/Rover_Main.ino` source encoding:** if you're restoring this file from an older export/copy, double-check that `<`, `>`, `"`, `/`, and `:` characters survived intact (`#include <Wire.h>`, comparison operators, string literals, `//` comments, and `switch` `case` colons are all easy to lose silently in a lossy copy-paste or PDF round-trip, and the sketch will fail to compile without throwing an obvious error in some editors).

## Troubleshooting

**Arduino sketch won't compile -- "Wire.h: No such file"**
→ Make sure the line reads `#include <Wire.h>` with angle brackets; a flattened/lossy copy of this file can silently drop them.

**HC-05 won't pair**
→ Default pairing code is usually `1234` or `0000`. Confirm baud rate matches `BT.begin(9600)` in the firmware.

**Voice app: "Vosk model not found"**
→ Run `bash rover_nlp_app/setup.sh` -- the model is not committed to git and must be downloaded once.

**Rover doesn't reverse when something's in front of it**
→ Check `OBSTACLE_DISTANCE` in the firmware (default 30cm) and confirm the HC-SR04 TRIG/ECHO pins (30/31) match your wiring.

**RL simulation: "Pygame not found" / no display (SSH/headless)**
→ `pip install pygame`, or run `python src/simulate.py --video` to save an MP4 instead of opening a live window.

## Authors

Sriramaneni Suhas · Sharuk · B. Jayanth · K. Jayanth · Ch. Rishi · S. Dinesh
Department of Computer Science & Engineering · VIT-AP University

## License

MIT -- see [LICENSE](LICENSE). The offline Vosk speech model is © Alpha Cephei Inc. under its own license -- see [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models).
