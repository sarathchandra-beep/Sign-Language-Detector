import os
import argparse
import glob
import cv2
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
import mediapipe as mp

mp_hands = mp.solutions.hands

def extract_hand_landmarks(image_bgr):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = hands.process(image_rgb)
        if not res.multi_hand_landmarks:
            return None
        lm = res.multi_hand_landmarks[0]
        pts = []
        for p in lm.landmark:
            pts.append([p.x, p.y, p.z])
        pts = np.array(pts).flatten()
        # normalize by subtracting wrist (idx 0) and dividing by max distance to make scale/translation invariant
        pts3 = pts.reshape(-1, 3)
        wrist = pts3[0]
        rel = pts3 - wrist
        scale = np.max(np.linalg.norm(rel, axis=1)) + 1e-6
        rel /= scale
        return rel.flatten()

def load_dataset(ds_dir):
    X, y = [], []
    ds_dir = Path(ds_dir)
    classes = sorted([d.name for d in ds_dir.iterdir() if d.is_dir()])
    for cls in classes:
        for fp in glob.glob(str(ds_dir/cls/"*")):
            img = cv2.imread(fp)
            if img is None:
                continue
            feat = extract_hand_landmarks(img)
            if feat is None:
                continue
            X.append(feat)
            y.append(cls)
    return np.array(X), np.array(y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="path to dataset with subfolders per label")
    ap.add_argument("--out", default="models", help="output dir for saved model")
    args = ap.parse_args()

    X, y = load_dataset(args.dataset)
    if len(X) == 0:
        raise SystemExit("No samples extracted. Check your dataset or landmark detection.")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True, C=10, gamma="scale", random_state=42))
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=le.classes_))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    dump(clf, outdir/"model.joblib")
    dump(le, outdir/"labels.joblib")
    print(f"Saved model to {outdir/'model.joblib'} and labels to {outdir/'labels.joblib'}")

if __name__ == "__main__":
    main()
