"""
CHIP — Offline Voice Rover Control
────────────────────────────────────────────────────────────────────
Speech   : Vosk  (100% offline, no internet needed)
NLP      : Conv1D CNN (TensorFlow / Keras)
Hardware : Arduino Mega + L298N + HC-05 via pyserial
────────────────────────────────────────────────────────────────────
SETUP (one-time):
  pip install streamlit tensorflow pyserial vosk pyaudio streamlit-autorefresh
  Download Vosk model → https://alphacephei.com/vosk/models
  Recommended : vosk-model-small-en-us-0.15  (40 MB)
  Extract folder to same directory as this app.py
  Rename folder to:  vosk-model
────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import numpy as np
import re
import time
import json
import threading
import queue
import os
from collections import Counter
from datetime import datetime

# ── Vosk offline STT ─────────────────────────────────────────────────────────
try:
    import vosk
    import pyaudio
    VOSK_OK = True
except ImportError:
    VOSK_OK = False

# ── Bluetooth serial ──────────────────────────────────────────────────────────
try:
    import serial
    import serial.tools.list_ports
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False

# ── Auto-refresh ──────────────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    REFRESH_OK = True
except ImportError:
    REFRESH_OK = False

# ── TensorFlow / Keras ───────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout, BatchNormalization,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
VOSK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "vosk-model")

WAKE_PHRASES = ["hey chip", "hey, chip", "hi chip", "hay chip", "a chip", "hey ship"]
STOP_PHRASES = ["bye chip", "bye, chip", "by chip", "goodbye chip", "stop chip", "bye ship"]

CMD_CHAR = {"FORWARD": "F", "BACKWARD": "B", "LEFT": "L", "RIGHT": "R", "STOP": "S"}

STATES = {
    "SLEEPING":   ("💤", "#334",    "SLEEPING",    "Waiting for  'Hey CHIP'"),
    "WAKING":     ("👂", "#ffcc00", "WAKING UP",   "Wake word detected!"),
    "LISTENING":  ("🎙️", "#00ff88", "LISTENING",   "Speak your command…"),
    "PROCESSING": ("⚡", "#39d0f5", "PROCESSING",  "Classifying…"),
}

SAMPLE_RATE   = 16000   # Vosk requires 16 kHz mono
CHUNK_SIZE    = 4000    # audio chunk size in frames

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + AUTO-REFRESH
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CHIP — Offline Rover", layout="wide", page_icon="🤖")

if REFRESH_OK:
    st_autorefresh(interval=700, key="chip_refresh")

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;500;700;900&display=swap');
html,body,[class*="css"]{font-family:'Barlow',sans-serif;background:#080a10;color:#c8d4e8;}
[data-testid="stSidebar"]{background:#0d0f18 !important;border-right:1px solid #1e2230;}
h1,h2,h3{font-family:'Share Tech Mono',monospace;color:#39d0f5;letter-spacing:2px;}
.stTextInput>div>div>input{background:#12151f;border:1px solid #39d0f540;color:#e8f0ff;
    border-radius:6px;font-family:'Share Tech Mono',monospace;font-size:1rem;}
.stTextInput>div>div>input:focus{border-color:#39d0f5;box-shadow:0 0 8px #39d0f544;}
.stButton>button{background:#12151f;border:1px solid #39d0f555;color:#39d0f5;
    font-family:'Share Tech Mono',monospace;border-radius:6px;transition:all 0.2s;width:100%;}
.stButton>button:hover{background:#39d0f522;border-color:#39d0f5;}
.stSelectbox>div>div{background:#12151f;border:1px solid #39d0f540;color:#c8d4e8;}
.metric-card{background:#12151f;border:1px solid #1e2230;border-radius:10px;
    padding:14px 18px;text-align:center;}
.metric-card .val{font-family:'Share Tech Mono',monospace;font-size:1.5rem;color:#39d0f5;}
.metric-card .lbl{font-size:0.7rem;color:#556;text-transform:uppercase;letter-spacing:1px;}
.status-panel{border-radius:14px;padding:22px 28px;text-align:center;margin-bottom:12px;}
.status-sleeping  {background:#0d0f18;border:1px solid #1e2230;}
.status-waking    {background:#261f00;border:1px solid #ffcc0044;box-shadow:0 0 20px #ffcc0022;}
.status-listening {background:#00261a;border:1px solid #00ff8844;box-shadow:0 0 20px #00ff8822;}
.status-processing{background:#001226;border:1px solid #39d0f544;box-shadow:0 0 20px #39d0f522;}
.status-icon {font-size:2.8rem;}
.status-label{font-family:'Share Tech Mono',monospace;font-size:1.1rem;margin-top:6px;}
.status-hint {font-size:0.78rem;color:#556;margin-top:4px;font-family:'Share Tech Mono',monospace;}
.cmd-display{border-radius:12px;padding:22px 20px;text-align:center;
    font-family:'Share Tech Mono',monospace;font-size:2.2rem;
    font-weight:900;letter-spacing:5px;margin:10px 0 4px 0;}
.char-badge{display:block;text-align:center;font-family:'Share Tech Mono',monospace;
    font-size:0.85rem;padding:4px 0;letter-spacing:2px;color:#556;}
.FORWARD {background:#00261a;color:#00ff88;border:1px solid #00ff8844;box-shadow:0 0 20px #00ff8818;}
.BACKWARD{background:#261200;color:#ff8c00;border:1px solid #ff8c0044;box-shadow:0 0 20px #ff8c0018;}
.LEFT    {background:#001226;color:#00b4ff;border:1px solid #00b4ff44;box-shadow:0 0 20px #00b4ff18;}
.RIGHT   {background:#12002a;color:#c070ff;border:1px solid #c070ff44;box-shadow:0 0 20px #c070ff18;}
.STOP    {background:#2a0000;color:#ff3c3c;border:1px solid #ff3c3c44;box-shadow:0 0 20px #ff3c3c18;}
.UNKNOWN {background:#12151f;color:#556;border:1px solid #1e2230;}
.heard-box{background:#0d0f18;border:1px solid #1e2230;border-radius:8px;
    padding:10px 14px;font-family:'Share Tech Mono',monospace;
    font-size:0.85rem;color:#39d0f5;margin-bottom:8px;}
.conf-bar-bg{background:#1a1d28;border-radius:4px;height:8px;margin-top:6px;overflow:hidden;}
.conf-bar-fg{height:8px;border-radius:4px;}
.history-row{display:flex;justify-content:space-between;align-items:center;
    padding:6px 10px;border-radius:6px;background:#12151f;border:1px solid #1e2230;
    margin-bottom:4px;font-family:'Share Tech Mono',monospace;font-size:0.76rem;}
.bt-on {color:#00ff88;font-family:'Share Tech Mono',monospace;font-size:0.85rem;}
.bt-off{color:#ff4444;font-family:'Share Tech Mono',monospace;font-size:0.85rem;}
.warn-box{background:#1a1200;border:1px solid #ffcc0044;border-radius:8px;
    padding:12px 16px;font-family:'Share Tech Mono',monospace;
    font-size:0.82rem;color:#ffcc00;line-height:1.8;}
.ok-box  {background:#001a0d;border:1px solid #00ff8844;border-radius:8px;
    padding:12px 16px;font-family:'Share Tech Mono',monospace;
    font-size:0.82rem;color:#00ff88;line-height:1.8;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
RAW_DATA = [
    # FORWARD
    ("move forward","FORWARD"),("go forward","FORWARD"),("forward","FORWARD"),
    ("move ahead","FORWARD"),("drive forward","FORWARD"),("advance","FORWARD"),
    ("go ahead","FORWARD"),("proceed forward","FORWARD"),("head forward","FORWARD"),
    ("keep going forward","FORWARD"),("move straight","FORWARD"),("go straight","FORWARD"),
    ("drive straight","FORWARD"),("straight ahead","FORWARD"),("continue forward","FORWARD"),
    ("continue ahead","FORWARD"),("keep moving forward","FORWARD"),("push forward","FORWARD"),
    ("full speed ahead","FORWARD"),("march forward","FORWARD"),("roll forward","FORWARD"),
    ("travel forward","FORWARD"),("speed forward","FORWARD"),("navigate forward","FORWARD"),
    ("move in front","FORWARD"),("step forward","FORWARD"),("inch forward","FORWARD"),
    ("forward motion","FORWARD"),("move north","FORWARD"),("go north","FORWARD"),
    ("charge forward","FORWARD"),("rush forward","FORWARD"),("surge forward","FORWARD"),
    ("creep forward","FORWARD"),("accelerate forward","FORWARD"),("go onward","FORWARD"),
    ("onward","FORWARD"),("drive onward","FORWARD"),("proceed","FORWARD"),
    ("keep straight","FORWARD"),("forward please","FORWARD"),("go straight ahead","FORWARD"),
    ("move it forward","FORWARD"),("lets go forward","FORWARD"),("head on forward","FORWARD"),
    ("keep going straight","FORWARD"),("continue straight","FORWARD"),("advance forward","FORWARD"),
    # BACKWARD
    ("move backward","BACKWARD"),("go back","BACKWARD"),("reverse","BACKWARD"),
    ("go backward","BACKWARD"),("drive backward","BACKWARD"),("back up","BACKWARD"),
    ("move back","BACKWARD"),("retreat","BACKWARD"),("go backwards","BACKWARD"),
    ("move backwards","BACKWARD"),("reverse direction","BACKWARD"),("back off","BACKWARD"),
    ("roll backward","BACKWARD"),("step backward","BACKWARD"),("move in reverse","BACKWARD"),
    ("go in reverse","BACKWARD"),("move south","BACKWARD"),("go south","BACKWARD"),
    ("head backward","BACKWARD"),("back away","BACKWARD"),("pull back","BACKWARD"),
    ("inch backward","BACKWARD"),("creep backward","BACKWARD"),("reverse slowly","BACKWARD"),
    ("reverse fast","BACKWARD"),("backward direction","BACKWARD"),("go to the back","BACKWARD"),
    ("reverse movement","BACKWARD"),("back it up","BACKWARD"),("go rearward","BACKWARD"),
    ("rearward","BACKWARD"),("drive back","BACKWARD"),("head south","BACKWARD"),
    ("keep going back","BACKWARD"),("continue backward","BACKWARD"),("move rearward","BACKWARD"),
    ("reverse course","BACKWARD"),("backward please","BACKWARD"),("lets go back","BACKWARD"),
    ("move it back","BACKWARD"),("accelerate backward","BACKWARD"),("take it back","BACKWARD"),
    ("push backward","BACKWARD"),("head back","BACKWARD"),("reverse the rover","BACKWARD"),
    # LEFT
    ("turn left","LEFT"),("go left","LEFT"),("left","LEFT"),("rotate left","LEFT"),
    ("move left","LEFT"),("steer left","LEFT"),("veer left","LEFT"),("bear left","LEFT"),
    ("swing left","LEFT"),("head left","LEFT"),("drive left","LEFT"),("take a left","LEFT"),
    ("turn to the left","LEFT"),("go to the left","LEFT"),("move to the left","LEFT"),
    ("navigate left","LEFT"),("shift left","LEFT"),("drift left","LEFT"),("curve left","LEFT"),
    ("go east","LEFT"),("head east","LEFT"),("left turn","LEFT"),("sharp left","LEFT"),
    ("slight left","LEFT"),("pivot left","LEFT"),("lean left","LEFT"),("bank left","LEFT"),
    ("angle left","LEFT"),("face left","LEFT"),("go counterclockwise","LEFT"),
    ("turn counterclockwise","LEFT"),("left side","LEFT"),("towards the left","LEFT"),
    ("left please","LEFT"),("take the left","LEFT"),("spin left","LEFT"),("wheel left","LEFT"),
    ("steer to the left","LEFT"),("turn the wheel left","LEFT"),("go left now","LEFT"),
    ("hard left","LEFT"),("soft left","LEFT"),("move to left side","LEFT"),
    ("navigate to the left","LEFT"),("keep turning left","LEFT"),("continue left","LEFT"),
    # RIGHT
    ("turn right","RIGHT"),("go right","RIGHT"),("right","RIGHT"),("rotate right","RIGHT"),
    ("move right","RIGHT"),("steer right","RIGHT"),("veer right","RIGHT"),("bear right","RIGHT"),
    ("swing right","RIGHT"),("head right","RIGHT"),("drive right","RIGHT"),("take a right","RIGHT"),
    ("turn to the right","RIGHT"),("go to the right","RIGHT"),("move to the right","RIGHT"),
    ("navigate right","RIGHT"),("shift right","RIGHT"),("drift right","RIGHT"),("curve right","RIGHT"),
    ("go west","RIGHT"),("head west","RIGHT"),("right turn","RIGHT"),("sharp right","RIGHT"),
    ("slight right","RIGHT"),("pivot right","RIGHT"),("lean right","RIGHT"),("bank right","RIGHT"),
    ("angle right","RIGHT"),("face right","RIGHT"),("go clockwise","RIGHT"),
    ("turn clockwise","RIGHT"),("right side","RIGHT"),("towards the right","RIGHT"),
    ("right please","RIGHT"),("take the right","RIGHT"),("spin right","RIGHT"),("wheel right","RIGHT"),
    ("steer to the right","RIGHT"),("turn the wheel right","RIGHT"),("go right now","RIGHT"),
    ("hard right","RIGHT"),("soft right","RIGHT"),("move to right side","RIGHT"),
    ("navigate to the right","RIGHT"),("keep turning right","RIGHT"),("continue right","RIGHT"),
    # STOP
    ("stop","STOP"),("halt","STOP"),("freeze","STOP"),("stop now","STOP"),
    ("halt immediately","STOP"),("stop moving","STOP"),("stay still","STOP"),
    ("do not move","STOP"),("emergency stop","STOP"),("brake","STOP"),
    ("full stop","STOP"),("hold position","STOP"),("hold still","STOP"),
    ("stand still","STOP"),("dont move","STOP"),("cease movement","STOP"),
    ("abort","STOP"),("cancel movement","STOP"),("stop the rover","STOP"),
    ("stay","STOP"),("pause","STOP"),("wait","STOP"),("stop right now","STOP"),
    ("cease","STOP"),("shut down","STOP"),("kill movement","STOP"),
    ("stop immediately","STOP"),("stop at once","STOP"),("freeze in place","STOP"),
    ("stop going","STOP"),("kill motion","STOP"),("cut movement","STOP"),
    ("no movement","STOP"),("end movement","STOP"),("stop all movement","STOP"),
    ("do not go","STOP"),("stop the vehicle","STOP"),("stop the robot","STOP"),
    ("hold on","STOP"),("stop please","STOP"),("please stop","STOP"),
    ("quit moving","STOP"),("stop motion","STOP"),("halt the rover","STOP"),
    ("engage brakes","STOP"),("apply brakes","STOP"),("cut engine","STOP"),
]

LABELS_LIST   = ["FORWARD","BACKWARD","LEFT","RIGHT","STOP"]
LABEL2IDX     = {l:i for i,l in enumerate(LABELS_LIST)}
COMMAND_EMOJI = {"FORWARD":"⬆️","BACKWARD":"⬇️","LEFT":"⬅️","RIGHT":"➡️","STOP":"🛑"}
CMD_COLOR     = {"FORWARD":"#00ff88","BACKWARD":"#ff8c00","LEFT":"#00b4ff","RIGHT":"#c070ff","STOP":"#ff3c3c"}
CONF_COLOR    = lambda c: "#00ff88" if c>=0.80 else "#ffcc00" if c>=0.55 else "#ff4444"

texts  = [x[0] for x in RAW_DATA]
labels = [LABEL2IDX[x[1]] for x in RAW_DATA]
MAX_WORDS, MAX_LEN = 400, 10

# ─────────────────────────────────────────────────────────────────────────────
# CNN MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Building CNN model…")
def build_cnn():
    tok = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    X = pad_sequences(tok.texts_to_sequences(texts), maxlen=MAX_LEN, padding="post")
    y = tf.keras.utils.to_categorical(labels, num_classes=5)
    model = Sequential([
        Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
        Conv1D(128, 3, activation="relu", padding="same"),
        BatchNormalization(),
        Conv1D(64,  2, activation="relu", padding="same"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(5, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=120, batch_size=16, verbose=0,
              callbacks=[EarlyStopping(patience=12, restore_best_weights=True)])
    return model, tok

cnn_model, tokenizer = build_cnn()

def nlp_predict(text: str):
    text = re.sub(r"[^a-z0-9 ]", "", text.lower().strip())
    seq  = pad_sequences(tokenizer.texts_to_sequences([text]),
                         maxlen=MAX_LEN, padding="post")
    prob = cnn_model.predict(seq, verbose=0)[0]
    idx  = int(np.argmax(prob))
    return LABELS_LIST[idx], float(prob[idx]), prob

# ─────────────────────────────────────────────────────────────────────────────
# VOSK MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📦 Loading Vosk offline model…")
def load_vosk_model():
    if not VOSK_OK:
        return None
    if not os.path.exists(VOSK_MODEL_PATH):
        return None
    vosk.SetLogLevel(-1)   # suppress verbose logs
    return vosk.Model(VOSK_MODEL_PATH)

vosk_model = load_vosk_model()

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "chip_state":  "SLEEPING",
    "chip_active": False,
    "stop_thread": False,
    "last_heard":  "",
    "last_cmd":    None,
    "history":     [],
    "bt_conn":     None,
    "total_preds": 0,
    "high_conf":   0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# BLUETOOTH
# ─────────────────────────────────────────────────────────────────────────────
def send_bt(command: str):
    char = CMD_CHAR.get(command, "S")
    conn = st.session_state.bt_conn
    if conn and conn.is_open:
        try:
            conn.write(char.encode())
            conn.flush()
            return True, char
        except Exception:
            st.session_state.bt_conn = None
    return False, char

def contains(text: str, phrases: list) -> bool:
    t = text.lower().strip()
    return any(p in t for p in phrases)

# ─────────────────────────────────────────────────────────────────────────────
# VOSK CONTINUOUS LISTENER THREAD
# ─────────────────────────────────────────────────────────────────────────────
def chip_listener_thread():
    """
    Uses Vosk (offline) to stream audio from mic.
    State machine: SLEEPING → LISTENING → SLEEPING
    Runs until stop_thread is set True.
    """
    if not VOSK_OK or vosk_model is None:
        return

    pa   = pyaudio.PyAudio()
    rec  = vosk.KaldiRecognizer(vosk_model, SAMPLE_RATE)

    stream = pa.open(
        format            = pyaudio.paInt16,
        channels          = 1,
        rate              = SAMPLE_RATE,
        input             = True,
        frames_per_buffer = CHUNK_SIZE,
    )
    stream.start_stream()

    try:
        while not st.session_state.get("stop_thread", False):
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text   = result.get("text", "").strip()
            else:
                partial = json.loads(rec.PartialResult())
                text    = partial.get("partial", "").strip()

            if not text:
                continue

            state = st.session_state.chip_state

            # ── SLEEPING: wait for wake word ──────────────────────────────────
            if state == "SLEEPING":
                if contains(text, WAKE_PHRASES):
                    st.session_state.last_heard = text
                    st.session_state.chip_state = "WAKING"
                    rec = vosk.KaldiRecognizer(vosk_model, SAMPLE_RATE)  # fresh
                    time.sleep(0.3)
                    st.session_state.chip_state = "LISTENING"

            # ── LISTENING: classify every utterance ───────────────────────────
            elif state == "LISTENING":
                # Only act on final results (not partials)
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text   = result.get("text", "").strip()
                    if not text:
                        continue

                    st.session_state.last_heard = text
                    st.session_state.chip_state = "PROCESSING"

                    # Stop word check
                    if contains(text, STOP_PHRASES):
                        send_bt("STOP")
                        st.session_state.last_cmd = {
                            "cmd":"— CHIP SLEEPING —","char":"S",
                            "conf":1.0,"heard":text,"bt":False,
                            "time":datetime.now().strftime("%H:%M:%S"),
                        }
                        rec = vosk.KaldiRecognizer(vosk_model, SAMPLE_RATE)
                        st.session_state.chip_state = "SLEEPING"
                        continue

                    # NLP classify
                    prediction, confidence, probs = nlp_predict(text)
                    bt_sent, char_sent = send_bt(prediction)

                    entry = {
                        "cmd":   prediction,
                        "char":  char_sent,
                        "conf":  confidence,
                        "probs": probs.tolist(),
                        "heard": text,
                        "bt":    bt_sent,
                        "time":  datetime.now().strftime("%H:%M:%S"),
                    }
                    st.session_state.last_cmd = entry
                    st.session_state.total_preds += 1
                    if confidence >= 0.80:
                        st.session_state.high_conf += 1
                    if confidence >= 0.45:
                        st.session_state.history.insert(0, entry)
                        if len(st.session_state.history) > 30:
                            st.session_state.history.pop()

                    st.session_state.chip_state = "LISTENING"

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        st.session_state.chip_active = False
        st.session_state.chip_state  = "SLEEPING"


# ─────────────────────────────────────────────────────────────────────────────
# THREAD CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
def start_chip():
    if st.session_state.chip_active:
        return
    st.session_state.stop_thread = False
    st.session_state.chip_state  = "SLEEPING"
    st.session_state.chip_active = True
    t = threading.Thread(target=chip_listener_thread, daemon=True)
    t.start()

def stop_chip():
    send_bt("STOP")
    st.session_state.stop_thread = True
    st.session_state.chip_active = False
    st.session_state.chip_state  = "SLEEPING"

# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def render_status():
    state = st.session_state.chip_state
    icon, color, label, hint = STATES.get(state, STATES["SLEEPING"])
    css = {"SLEEPING":"sleeping","WAKING":"waking",
           "LISTENING":"listening","PROCESSING":"processing"}.get(state,"sleeping")
    st.markdown(f"""
    <div class="status-panel status-{css}">
        <div class="status-icon">{icon}</div>
        <div class="status-label" style="color:{color};">{label}</div>
        <div class="status-hint">{hint}</div>
    </div>""", unsafe_allow_html=True)

def render_last_command():
    e = st.session_state.last_cmd
    if not e:
        st.markdown('<div style="color:#334;font-family:\'Share Tech Mono\';'
                    'font-size:0.82rem;text-align:center;padding:20px;">'
                    'No command yet…<br>Say <span style="color:#39d0f5;">Hey CHIP</span>'
                    ' to start</div>', unsafe_allow_html=True)
        return

    cmd   = e["cmd"]
    conf  = e["conf"]
    char  = e.get("char","—")
    heard = e.get("heard","")
    bt    = e.get("bt", False)

    if cmd in COMMAND_EMOJI:
        css   = cmd
        emoji = COMMAND_EMOJI[cmd]
        color = CMD_COLOR[cmd]
        st.markdown(f'<div class="cmd-display {css}">{emoji}&nbsp;&nbsp;{cmd}</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="char-badge">sends &nbsp;'
            f'<span style="color:{color};">\'{char}\'</span>'
            f'&nbsp; to Arduino {"📡" if bt else ""}</div>',
            unsafe_allow_html=True)
        bar_c = CONF_COLOR(conf)
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;
                    font-family:'Share Tech Mono';font-size:0.76rem;
                    color:#556;margin-top:10px;">
            <span>CONFIDENCE</span>
            <span style="color:{bar_c};">{conf*100:.1f}%</span>
        </div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fg" style="width:{conf*100:.1f}%;background:{bar_c};"></div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="cmd-display UNKNOWN" style="font-size:1.2rem;">{cmd}</div>',
            unsafe_allow_html=True)

    if heard:
        st.markdown(f'<div class="heard-box">🗣️ &nbsp;"{heard}"</div>',
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 HC-05 BLUETOOTH")
    st.divider()

    if SERIAL_OK:
        port_list = [p.device for p in serial.tools.list_ports.comports()]
        if not port_list:
            st.warning("No COM ports.\nPair HC-05 first.")
            port_list = ["—"]
        sel_port = st.selectbox("COM Port", port_list)
        baud     = st.selectbox("Baud Rate", [9600, 38400, 115200], index=0)
        c1, c2   = st.columns(2)
        if c1.button("🔗 Connect", use_container_width=True):
            try:
                st.session_state.bt_conn = serial.Serial(sel_port, baud, timeout=1)
                st.success(f"✅ {sel_port} @ {baud}")
            except Exception as e:
                st.error(str(e))
        if c2.button("✂️ Disconnect", use_container_width=True):
            if st.session_state.bt_conn:
                st.session_state.bt_conn.close()
                st.session_state.bt_conn = None
        connected = st.session_state.bt_conn is not None and st.session_state.bt_conn.is_open
        st.markdown(
            f'<div class="{"bt-on" if connected else "bt-off"}">'
            f'● {"CONNECTED ("+sel_port+")" if connected else "NOT CONNECTED"}</div>',
            unsafe_allow_html=True)
    else:
        st.code("pip install pyserial")

    st.divider()
    st.markdown("## 🔧 SYSTEM CHECK")

    vosk_exists = os.path.exists(VOSK_MODEL_PATH)
    items = [
        ("Vosk library",    VOSK_OK,       "pip install vosk"),
        ("PyAudio",         VOSK_OK,       "pipwin install pyaudio"),
        ("Vosk model",      vosk_exists,   "Download vosk-model → see steps"),
        ("pyserial",        SERIAL_OK,     "pip install pyserial"),
        ("Auto-refresh",    REFRESH_OK,    "pip install streamlit-autorefresh"),
    ]
    for name, ok, fix in items:
        icon = "✅" if ok else "❌"
        hint = "" if ok else f'<span style="color:#556;font-size:0.7rem;"> → {fix}</span>'
        st.markdown(
            f'<div style="font-family:\'Share Tech Mono\';font-size:0.78rem;'
            f'margin-bottom:4px;">{icon} {name}{hint}</div>',
            unsafe_allow_html=True)

    st.divider()
    st.markdown("## 🗣️ WAKE WORDS")
    st.markdown("""
    <div style="font-family:'Share Tech Mono';font-size:0.8rem;line-height:2.2;color:#556;">
    Start &nbsp;<span style="color:#00ff88;">"Hey CHIP"</span><br>
    Stop &nbsp;&nbsp;<span style="color:#ff3c3c;">"Bye CHIP"</span>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("## 📈 STATS")
    total = st.session_state.total_preds
    hc    = st.session_state.high_conf
    rate  = f"{hc/total*100:.0f}%" if total else "—"
    ca, cb = st.columns(2)
    ca.markdown(f'<div class="metric-card"><div class="val">{total}</div>'
                f'<div class="lbl">Commands</div></div>', unsafe_allow_html=True)
    cb.markdown(f'<div class="metric-card"><div class="val">{rate}</div>'
                f'<div class="lbl">High Conf</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🤖 CHIP — Offline Voice Rover")
st.caption("100% offline · Vosk STT · Conv1D CNN · HC-05 Bluetooth")
st.divider()

# ── Dependency warnings ───────────────────────────────────────────────────────
if not VOSK_OK:
    st.markdown('<div class="warn-box">❌ Vosk not installed.<br>'
                'Run: <b>pip install vosk</b></div>', unsafe_allow_html=True)
elif not os.path.exists(VOSK_MODEL_PATH):
    st.markdown(
        '<div class="warn-box">❌ Vosk model not found at: <b>./vosk-model/</b><br>'
        '1. Download from https://alphacephei.com/vosk/models<br>'
        '2. Get <b>vosk-model-small-en-us-0.15</b> (40 MB)<br>'
        '3. Extract ZIP → rename folder to <b>vosk-model</b><br>'
        '4. Place it in the same folder as app.py</div>',
        unsafe_allow_html=True)
else:
    st.markdown('<div class="ok-box">✅ Vosk model loaded — fully offline</div>',
                unsafe_allow_html=True)

st.markdown("")

# ── Controls ──────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([2, 2, 3])
ready = VOSK_OK and os.path.exists(VOSK_MODEL_PATH)

with c1:
    if not st.session_state.chip_active:
        if st.button("🟢 Start CHIP", use_container_width=True, disabled=not ready):
            start_chip()
            st.rerun()
    else:
        if st.button("🔴 Stop CHIP", use_container_width=True):
            stop_chip()
            st.rerun()

with c2:
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history  = []
        st.session_state.last_cmd = None
        st.rerun()

with c3:
    color  = "#00ff88" if st.session_state.chip_active else "#334"
    label  = "🟢 ACTIVE — listening for 'Hey CHIP'" if st.session_state.chip_active \
             else "⚫ INACTIVE — press Start CHIP"
    st.markdown(
        f'<div style="font-family:\'Share Tech Mono\';font-size:0.82rem;'
        f'color:{color};padding:8px 0;">{label}</div>',
        unsafe_allow_html=True)

if not REFRESH_OK:
    st.warning("Install `streamlit-autorefresh` for live updates: `pip install streamlit-autorefresh`")

st.markdown("")

# ── Main panels ───────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    st.markdown("### 📡 CHIP Status")
    render_status()

    st.markdown("### ⚡ Last Command")
    render_last_command()

    st.markdown("### ⌨️ Manual Override")
    st.caption("Type a command (works offline, no mic needed)")
    manual = st.text_input("m", placeholder='"go forward", "turn left", "stop"…',
                           label_visibility="collapsed")
    if st.button("⚡ Send Manual Command", use_container_width=True):
        if manual.strip():
            pred, conf, probs = nlp_predict(manual)
            bt_sent, char_sent = send_bt(pred)
            e = {
                "cmd":pred,"char":char_sent,"conf":conf,
                "probs":probs.tolist(),"heard":manual,
                "bt":bt_sent,"time":datetime.now().strftime("%H:%M:%S"),
            }
            st.session_state.last_cmd = e
            st.session_state.total_preds += 1
            if conf >= 0.80: st.session_state.high_conf += 1
            if conf >= 0.45:
                st.session_state.history.insert(0, e)
            st.rerun()

with right_col:
    st.markdown("### 📋 Command History")
    if not st.session_state.history:
        st.markdown('<div style="color:#334;font-family:\'Share Tech Mono\';'
                    'font-size:0.82rem;">No commands yet…</div>',
                    unsafe_allow_html=True)
    else:
        for h in st.session_state.history[:15]:
            cmd   = h.get("cmd","?")
            emoji = COMMAND_EMOJI.get(cmd,"❓")
            color = CONF_COLOR(h["conf"])
            bt_i  = f"📡'{h['char']}'" if h["bt"] else "—"
            heard = h.get("heard","")[:18]
            ts    = h.get("time","")
            st.markdown(f"""
            <div class="history-row">
                <span style="color:#334;">{ts}</span>
                <span style="color:#aaa;max-width:80px;overflow:hidden;
                    text-overflow:ellipsis;white-space:nowrap;">{heard}</span>
                <span style="color:{color};">{emoji} {cmd}</span>
                <span style="color:{color};">{h['conf']*100:.0f}%</span>
                <span style="color:#39d0f5;font-size:0.7rem;">{bt_i}</span>
            </div>""", unsafe_allow_html=True)

st.divider()
counts = Counter([x[1] for x in RAW_DATA])
with st.expander("📚 Dataset Breakdown"):
    cols = st.columns(5)
    for i, lbl in enumerate(LABELS_LIST):
        cols[i].markdown(
            f'<div class="metric-card"><div class="val">{counts[lbl]}</div>'
            f'<div class="lbl">{COMMAND_EMOJI[lbl]} {lbl}</div></div>',
            unsafe_allow_html=True)
    st.caption(f"Total training samples: {len(RAW_DATA)}")
