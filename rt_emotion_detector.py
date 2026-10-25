from collections import deque

import cv2
import joblib
import numpy as np
from skimage.feature import hog

model = joblib.load("./models/svm_hog_fer2013.joblib")

emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

# Smoothing: average the SVM decision scores of the last x frames
history = deque(maxlen=30)

def extract_hog(face):
    # Same settings as training: input must be a uint8 48x48 image in the 0-255 range
    return hog(face, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        history.clear()
    else:
        # Largest face only, so the smoothing follows one person
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Tight Haar crop, as in the provided demo; FER2013 faces are cropped closely as well
        face = cv2.resize(gray[y:y + h, x:x + w], (48, 48), interpolation=cv2.INTER_AREA)

        # One-vs-rest scores: pairwise votes per emotion plus a confidence tie-breaker
        history.append(model.decision_function(extract_hog(face).reshape(1, -1))[0])

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        scores = np.mean(history, axis=0)
        best = int(np.argmax(scores))
        cv2.putText(frame, emotions[best], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Smoothed share of pairwise duels won per emotion (6 duels per emotion)
        for i, (name, s) in enumerate(zip(emotions, scores)):
            top = 20 + 22 * i
            share = float(np.clip(s / (len(emotions) - 1), 0, 1))
            color = (0, 200, 200) if i == best else (200, 200, 200)
            
            cv2.putText(frame, f"{name} {share:.2f}", (10, top + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.rectangle(frame, (110, top), (110 + int(150 * share), top + 14), color, -1)

    cv2.imshow('Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
