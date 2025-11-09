import streamlit as st
import cv2
import numpy as np
from joblib import load
from pathlib import Path
import mediapipe as mp
from datetime import datetime, time
import pytz

st.set_page_config(page_title="Sign Language Detector", page_icon="🤟", layout="wide")

# --- Config ---
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "model.joblib"
LABELS_PATH = MODELS_DIR / "labels.joblib"
TZ = pytz.timezone("Asia/Kolkata")
ALLOWED_HOURS_IST = (time(0,0,0), time(23,59,59))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return None, None
    model = load(MODEL_PATH)
    labels = load(LABELS_PATH)
    return model, labels

def now_ist():
    return datetime.now(TZ).timetz()

def within_allowed_hours():
    start, end = ALLOWED_HOURS_IST
    t = now_ist()
    # simple range check; assumes start < end within same day
    return (t >= start) and (t <= end)

def extract_hand_landmarks_from_bgr(image_bgr):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)
        if not res.multi_hand_landmarks:
            return None, None
        lm = res.multi_hand_landmarks[0]
        pts = []
        for p in lm.landmark:
            pts.append([p.x, p.y, p.z])
        pts = np.array(pts).flatten()
        pts3 = pts.reshape(-1,3)
        wrist = pts3[0]
        rel = pts3 - wrist
        scale = np.max(np.linalg.norm(rel, axis=1)) + 1e-6
        rel /= scale
        return rel.flatten(), lm

def predict_bgr(image_bgr, model, labels):
    feats, lm = extract_hand_landmarks_from_bgr(image_bgr)
    if feats is None:
        return None, None, None
    probs = model.predict_proba([feats])[0]
    idx = int(np.argmax(probs))
    return labels.inverse_transform([idx])[0], float(probs[idx]), lm

st.title("🤟 Sign Language Detector")
st.caption("Upload an image or use your webcam. Operates only between 6 PM and 10 PM (Asia/Kolkata).")

open_now = within_allowed_hours()
if not open_now:
    st.error("⛔ The system is available only between 6 PM and 10 PM (Asia/Kolkata). Please come back during that window.")
    st.stop()

model, labels = load_assets()
if model is None:
    st.warning("Model not found. Train it first with `python train.py --dataset dataset --out models`.")
    st.stop()

tab1, tab2 = st.tabs(["📷 Upload Image", "🎥 Real-time Video"])

with tab1:
    up = st.file_uploader("Upload a hand image (JPG/PNG)", type=["jpg","jpeg","png"])
    col1, col2 = st.columns([1,1])
    if up:
        file_bytes = np.frombuffer(up.read(), np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        pred, conf, lm = predict_bgr(bgr, model, labels)
        with col1:
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Uploaded")
        with col2:
            if pred is None:
                st.error("No hand landmarks detected.")
            else:
                st.success(f"Prediction: **{pred}** (confidence {conf:.2f})")
                # draw landmarks
                img_draw = bgr.copy()
                img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
                if lm:
                    mp_drawing.draw_landmarks(img_rgb, lm, mp_hands.HAND_CONNECTIONS)
                st.image(img_rgb, caption="Landmarks")

with tab2:
    run = st.checkbox("Start camera")
    info = st.empty()
    frame_holder = st.empty()
    if run:
        cap = cv2.VideoCapture(0)
        with mp_hands.Hands(static_image_mode=False, max_num_hands=1) as hands:
            while run:
                ok, frame = cap.read()
                if not ok:
                    info.error("Could not read from camera.")
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)
                pred_text = ""
                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0]
                    # extract features from landmarks
                    pts = []
                    for p in lm.landmark:
                        pts.append([p.x, p.y, p.z])
                    pts = np.array(pts).reshape(-1,3)
                    wrist = pts[0]
                    rel = pts - wrist
                    scale = np.max(np.linalg.norm(rel, axis=1)) + 1e-6
                    rel /= scale
                    feats = rel.flatten()
                    probs = model.predict_proba([feats])[0]
                    idx = int(np.argmax(probs))
                    label = labels.inverse_transform([idx])[0]
                    conf = float(probs[idx])
                    pred_text = f"{label} ({conf:.2f})"
                    mp_drawing.draw_landmarks(rgb, lm, mp_hands.HAND_CONNECTIONS)
                cv2.putText(rgb, pred_text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
                frame_holder.image(rgb)
                run = st.session_state.get('Start camera', True)
        cap.release()
