import pickle
from collections import deque

import cv2
import dlib
import numpy as np
import skfuzzy as fuzz

with open("./models/fuzzy_kdef_config.pkl", "rb") as f:
    bundle = pickle.load(f)

feature_names = bundle["feature_names"]
input_sets = bundle["input_sets"]
rules = bundle["rules"]
x_support = bundle["output_range"]
y_support_low, y_support_high = bundle["output_low"], bundle["output_high"]
emotions = bundle["emotion_order"]
fallback = bundle["no_rule_fallback"]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
landmark_predictor = dlib.shape_predictor("faceAndLandmarkModels/shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)

# Smoothing: average the fuzzy scores and rule strengths of the last x frames
history = deque(maxlen=60)

def extract_landmarks(face):
    # Crop whole face, so the face rectangle is the full image
    height, width = face.shape
    shape = landmark_predictor(face, dlib.rectangle(0, 0, width - 1, height - 1))
    return np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float32)

def geometric_features(points):
    # Copy of geometric_features in Task4.ipynb; the order must match feature_names
    def distance(a, b):
        return np.linalg.norm(a - b)

    eye_left = points[36:42].mean(axis=0)
    eye_right = points[42:48].mean(axis=0)

    # Rotate so the eye centres lie horizontally (reduces head tilt)
    eye_vector = eye_right - eye_left
    scale = np.linalg.norm(eye_vector)
    if scale < 1e-6:
        raise ValueError("Eye centres are too close")

    horizontal = eye_vector / scale
    vertical = np.array([-horizontal[1], horizontal[0]])
    centred = points - (eye_left + eye_right) / 2
    p = np.column_stack([centred @ horizontal, centred @ vertical])

    mouth_opening = np.mean([distance(p[61], p[67]), distance(p[62], p[66]), distance(p[63], p[65])]) / scale
    mouth_width = distance(p[48], p[54]) / scale
    
    if mouth_width < 1e-6:
        raise ValueError("Invalid mouth width")
    mouth_aspect_ratio = mouth_opening / mouth_width

    eye_opening = np.mean([distance(p[37], p[41]), distance(p[38], p[40]),
                           distance(p[43], p[47]), distance(p[44], p[46])]) / scale
    
    eyebrow_height = np.mean([p[36:42, 1].mean() - p[17:22, 1].mean(),
                              p[42:48, 1].mean() - p[22:27, 1].mean()]) / scale
    
    brow_separation = distance(p[21], p[22]) / scale

    lip_centre_y = (p[51, 1] + p[57, 1]) / 2
    corners_y = (p[48, 1] + p[54, 1]) / 2
    smile_curvature = (lip_centre_y - corners_y) / scale

    inner_brow_elevation = np.mean([p[17, 1] - p[21, 1], p[26, 1] - p[22, 1]]) / scale

    return np.array([mouth_opening, mouth_width, eye_opening, eyebrow_height,
                     brow_separation, smile_curvature, mouth_aspect_ratio, inner_brow_elevation])


def fuzzy_scores(feature_values):
    # Fuzzify, clipping to the training range 
    memberships = {}
    for feature, value in zip(feature_names, feature_values):
        sets = input_sets[feature]
        value = np.clip(value, sets["range"][0], sets["range"][-1])
        memberships[feature] = {term: float(fuzz.interp_membership(sets["range"], sets[term], value))
                                for term in ["low", "medium", "high"]}

    scores, strengths = [], []
    for emotion in emotions:
        # AND = min within a rule, OR = max across rules
        high = max(min(memberships[f][t] for f, t in conditions) for conditions in rules[emotion])
        aggregated = np.fmax(np.fmin(1.0 - high, y_support_low), np.fmin(high, y_support_high))
        scores.append(fuzz.defuzz(x_support, aggregated, "centroid"))
        strengths.append(high)

    return np.array(scores), np.array(strengths)

def decide(scores, strengths):
    if strengths.max() <= 1e-12:
        return fallback
    winners = [e for e, s in zip(emotions, scores) if np.isclose(s, scores.max(), rtol=0, atol=1e-10)]
    return fallback if fallback in winners else winners[0]

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

        # Crop to the Haar box and resize to the 48x48 format the fuzzy sets were fitted on
        face = cv2.resize(gray[y:y + h, x:x + w], (48, 48), interpolation=cv2.INTER_AREA)

        try:
            history.append(fuzzy_scores(geometric_features(extract_landmarks(face))))
        except ValueError:
            pass  # Degenerate landmarks: keep the previous history

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if history:
            scores = np.mean([s for s, _ in history], axis=0)
            strengths = np.mean([r for _, r in history], axis=0)
            label = decide(scores, strengths)
            cv2.putText(frame, label.capitalize(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Smoothed rule strength (degree of "high support") per emotion
            best = emotions.index(label)
            for i, (name, s) in enumerate(zip(emotions, strengths)):
                top = 20 + 22 * i
                color = (0, 200, 200) if i == best else (200, 200, 200)
                cv2.putText(frame, f"{name.capitalize()} {s:.2f}", (10, top + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                cv2.rectangle(frame, (110, top), (110 + int(150 * s), top + 14), color, -1)

    cv2.imshow('Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
