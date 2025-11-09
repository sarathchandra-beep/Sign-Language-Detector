# Sign Language Detection (Keypoint-based)

This project trains a simple classifier on hand landmarks (21×3) extracted by MediaPipe Hands
and serves a Streamlit GUI that supports both **image upload** and **real-time video**.

## Features
- Train on your own images organized as `dataset/<label>/*.jpg` (or .png).
- Extract 3D hand landmarks, normalize, and train an SVC classifier.
- Streamlit app with:
  - Upload Image
  - Real-time Video (OpenCV)
  - Works only between **6 PM and 10 PM Asia/Kolkata** (configurable).
- Lightweight and fast on CPU.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Prepare dataset
# dataset/
#   HELLO/ img1.jpg img2.jpg ...
#   THANKYOU/ ...
#   YES/ ...
#   NO/ ...

# 2) Train
python train.py --dataset dataset --out models

# 3) Run app
streamlit run app.py
```

## Notes
- If landmarks aren't detected for an image/frame, prediction is skipped.
- You can change the allowed hours in `app.py` (search for ALLOWED_HOURS_IST).
