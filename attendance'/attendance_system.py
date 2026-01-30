"""
Automatic Attendance System using DeepFace (No dlib required)

Dependencies:
- pip install deepface opencv-python pandas numpy

This script:
- Allows adding new students (capturing multiple images)
- Builds embeddings for each student using DeepFace
- Marks attendance via real-time webcam feed
- Saves data in:
  - students/ : student images
  - encodings.pkl : pickled dict {name: embedding}
  - attendance/YYYY-MM-DD.csv : daily attendance

"""

import os
import cv2
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, date
from deepface import DeepFace
import uuid

STUDENTS_DIR = "students"
ENCODINGS_FILE = "embeddings.pkl"
ATTENDANCE_DIR = "attendance"
UNKNOWN_DIR = "unknowns"
CAPTURE_COUNT = 5
TOLERANCE = 0.5

os.makedirs(STUDENTS_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

def sanitize_name(name: str) -> str:
    return "_".join(name.strip().split())

def capture_images_for_student(name: str):
    safe = sanitize_name(name)
    sid = safe + "_" + uuid.uuid4().hex[:6]
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print(f"[INFO] Capturing {CAPTURE_COUNT} images for {name}. Press SPACE to capture, ESC to cancel.")
    count = 0
    while count < CAPTURE_COUNT:
        ret, frame = cap.read()
        if not ret:
            continue
        display = frame.copy()
        h, w = display.shape[:2]
        cv2.putText(display, f"{name} - {count}/{CAPTURE_COUNT}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Capture", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            filename = os.path.join(STUDENTS_DIR, f"{sid}_{count+1}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[INFO] Saved {filename}")
            count += 1
    cap.release()
    cv2.destroyAllWindows()


def build_embeddings():
    print("[INFO] Building DeepFace embeddings...")
    files = [f for f in os.listdir(STUDENTS_DIR) if f.lower().endswith(('jpg','jpeg','png'))]
    if not files:
        print("[WARN] No images found in students/.")
        return

    groups = {}
    for f in files:
        prefix = f.rsplit("_", 1)[0]
        groups.setdefault(prefix, []).append(os.path.join(STUDENTS_DIR, f))

    embeddings = {}
    for prefix, imgpaths in groups.items():
        name = prefix.rsplit("_", 1)[0]
        vecs = []
        for p in imgpaths:
            try:
                rep = DeepFace.represent(p, model_name='Facenet512', enforce_detection=False)
                if rep:
                    vecs.append(rep[0]['embedding'])
            except Exception as e:
                print(f"[WARN] Error embedding {p}: {e}")
        if vecs:
            avg = np.mean(vecs, axis=0)
            embeddings[name] = avg
            print(f"[INFO] Built embedding for {name}")

    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"[INFO] Embeddings saved to {ENCODINGS_FILE}")


def load_embeddings():
    if not os.path.exists(ENCODINGS_FILE):
        raise FileNotFoundError("Embeddings file missing. Build embeddings first.")
    with open(ENCODINGS_FILE, 'rb') as f:
        data = pickle.load(f)
    return data


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def mark_attendance():
    embeddings = load_embeddings()
    today = date.today().isoformat()
    out_csv = os.path.join(ATTENDANCE_DIR, f"{today}.csv")

    # Ensure the CSV has the new Date column
    if not os.path.exists(out_csv):
        pd.DataFrame(columns=['Name', 'Date', 'Time']).to_csv(out_csv, index=False)

    df_att = pd.read_csv(out_csv)
    marked = set(df_att['Name'].tolist())

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Starting attendance. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            faces = DeepFace.extract_faces(frame, enforce_detection=False)
        except Exception as e:
            print(f"[WARN] Error detecting faces: {e}")
            continue

        for f in faces:
            area = f.get('facial_area')
            if not area:
                continue

            try:
                x, y, w, h = (
                    int(area.get('x', 0)),
                    int(area.get('y', 0)),
                    int(area.get('w', 0)),
                    int(area.get('h', 0)),
                )
            except (TypeError, ValueError):
                continue

            if w == 0 or h == 0:
                continue

            crop = frame[y:y + h, x:x + w]
            rep = DeepFace.represent(crop, model_name='Facenet512', enforce_detection=False)
            if not rep:
                continue

            emb = rep[0]['embedding']
            best_name = 'Unknown'
            best_score = -1

            for name, known_emb in embeddings.items():
                score = cosine_similarity(emb, known_emb)
                if score > best_score:
                    best_score = score
                    best_name = name

            if best_score > (1 - TOLERANCE):
                if best_name not in marked:
                    now_time = datetime.now().strftime('%H:%M:%S')
                    today_date = date.today().strftime('%Y-%m-%d')
                    pd.DataFrame([[best_name, today_date, now_time]],
                                 columns=['Name', 'Date', 'Time']).to_csv(
                        out_csv, mode='a', header=False, index=False
                    )
                    marked.add(best_name)
                    print(f"[INFO] Marked {best_name} on {today_date} at {now_time}")
            else:
                best_name = 'Unknown'

            color = (0, 255, 0) if best_name != 'Unknown' else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, best_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Attendance', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Attendance saved to {out_csv}")


def main_menu():
    while True:
        print('\n===== Automatic Attendance (DeepFace) =====')
        print('1) Add new student')
        print('2) Build embeddings')
        print('3) Start attendance')
        print('4) Exit')
        ch = input('Enter choice: ').strip()
        if ch == '1':
            name = input('Enter student name: ').strip()
            if name:
                capture_images_for_student(name)
        elif ch == '2':
            build_embeddings()
        elif ch == '3':
            mark_attendance()
        elif ch == '4':
            break
        else:
            print('Invalid option.')

if __name__ == '__main__':
    main_menu()
