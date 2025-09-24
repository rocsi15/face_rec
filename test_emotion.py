import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json

# Define global variables
data_path = "data/test"  # Replace with the actual path to your test images folder
image_size = (48, 48)
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, image_size)
        img = np.expand_dims(img, -1)  # Add channel dimension
        img = np.expand_dims(img, 0)  # Add batch dimension
        img = img / 255.0  # Normalize the image
    return img

def predict_emotion(image_path, model):
    img = load_and_preprocess_image(image_path)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    emotion = emotions[predicted_class[0]]
    return emotion

def display_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def process_test_images(data_path, model, output_file):
    results = {emotion: [] for emotion in emotions}

    for emotion in emotions:
        emotion_folder = os.path.join(data_path, emotion)
        if not os.path.isdir(emotion_folder):
            continue
        for filename in os.listdir(emotion_folder):
            image_path = os.path.join(emotion_folder, filename)
            predicted_emotion = predict_emotion(image_path, model)
            if predicted_emotion == emotion:
                results[emotion].append(filename)
                # display_image(image_path)
    
    with open(output_file, 'w') as f:
        for emotion in emotions:
            f.write(f"{emotion}:\n")
            for filename in results[emotion]:
                f.write(f"  {filename}\n")
            f.write("\n")

def main():

    print('c')
    # Load the model architecture from JSON file
    with open('emotion_recognition_architecture_custom_final.json', 'r') as json_file:
        model_json = json_file.read()
    print('b')
    model = model_from_json(model_json)
    print('d')
    # Load the weights from HDF5 file
    model.load_weights('emotion_recognition_model_custom_weights_final.h5')
    print('e')
    process_test_images(data_path, model, 'emotion.txt')
    print('f')

    
if __name__ == "__main__":
    main()