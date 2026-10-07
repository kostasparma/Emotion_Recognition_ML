import cv2
import numpy as np
from skimage.feature import hog
import joblib  # for loading the trained SVM model
import dlib 

# Load your trained SVM model
model = joblib.load("./models/rf_fer_model_landmarks.pkl")

# Importing dlib's landmark detector
# Dlib landmarks from: 
#   https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2
landmDet = dlib.shape_predictor("./faceAndLandmarkModels/shape_predictor_68_face_landmarks_GTX.dat")
#faceDetector = dlib.get_frontal_face_fetector()

def extractLandmarksFER(img):
    face_rects = dlib.rectangle(left=0, top=0, right=47, bottom=47)
    la = landmDet(img.astype(np.uint8), face_rects)
    x1 = face_rects.left() 
    y1 = face_rects.top() 
    x2 = face_rects.right() 
    y2 = face_rects.bottom()

    la2 = np.asarray( np.matrix([
        [
            (p.x-0.5*(x2-x1))/(x2-x1), (p.y-0.5*(y2-y1))/(y2-y1)
        ] for p in la.parts()])).astype(np.float32)
    la2 -= np.mean(la2, axis=0, keepdims=True)
    la2 /= np.max(np.abs(la2), axis=0, keepdims=True)
    
    return(la2.flatten())


# Emotion labels (adjust if needed)
emotion_labels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral']

# Initialize webcam
cap = cv2.VideoCapture(0)

# The students have to download the file haarcascade_frontalface_default.xml
# from Open CV. This is needed for the Haar cascade, which detects faces. This is a function built
# in Open CV (CascadeClassifier). However, it needs the XML file as input.
face_cascade = cv2.CascadeClassifier('faceAndLandmarkModels/haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    # The captured faces are resized to 48x48
    #  and HOG features are captured on top.
    # Finally, the classifier is evaluated
    for (x, y, w, h) in faces:
        # Extract face ROI
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))

        # Extract features
        features = extractLandmarksFER(face_resized)
        features = features.reshape(1, -1)

        # Predict emotion
        prediction = model.predict(features)
        label = emotion_labels[int(prediction)]

        # Display label
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

