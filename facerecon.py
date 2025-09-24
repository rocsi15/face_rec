import cv2
import numpy as np
from keras.models import load_model, model_from_json

# Load the model architecture from JSON file
with open('emotion_recognition_architecture_custom_final.json', 'r') as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# Load the weights from HDF5 file
model.load_weights('emotion_recognition_model_custom_weights_final.h5')

# Define emotions
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = np.expand_dims(face, axis=-1)  # Add channel dimension
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    face = face / 255.0  # Normalize the image
    return face

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face from the frame
        face = frame[y:y+h, x:x+w]

        # Preprocess the face
        preprocessed_face = preprocess_face(face)

        # Predict emotion
        predictions = model.predict(preprocessed_face)
        predicted_class = np.argmax(predictions, axis=1)
        emotion = emotions[predicted_class[0]]

        # Draw a rectangle around the face and put the emotion label
        color = (255, 105, 180)  # Pink color
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 10)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()