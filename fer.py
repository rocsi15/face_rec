import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau   
from sklearn.preprocessing import LabelBinarizer

# Define global variables
data_path = "data"
image_size = (48, 48)
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_images_from_folder(folder):
    images = []
    labels = []
    for emotion in emotions:
        emotion_folder = os.path.join(folder, emotion)
        for filename in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, image_size)
                img = np.expand_dims(img, -1)  # Add channel dimension
                images.append(img)
                labels.append(emotion)
    return images, labels

def create_enhanced_custom_model():
    model = Sequential()

    # First block
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Second block
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Third block
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    # Fourth block
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    # Global Average Pooling
    model.add(GlobalAveragePooling2D())

    # Fully connected layer
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(len(emotions), activation='softmax'))

    return model

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, image_size)
        img = np.expand_dims(img, -1)  # Add channel dimension
        img = np.expand_dims(img, 0)  # Add batch dimension
        img = img / 255.0  # Normalize the image
    return img

def display_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def predict_emotion(image_path, model):
    img = load_and_preprocess_image(image_path)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    emotion = emotions[predicted_class[0]]
    return emotion

def train_model(model, train_generator, val_generator, target_accuracy=0.90, max_epochs=3):
    # Calculate steps_per_epoch and validation_steps
    steps_per_epoch = train_generator.n // train_generator.batch_size
    validation_steps = val_generator.n // val_generator.batch_size
    
    current_epoch = 0
    best_val_accuracy = 0.0

    while current_epoch < max_epochs and best_val_accuracy < target_accuracy:
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(f'best_model_custom_epoch_{current_epoch}.keras', monitor='val_accuracy', save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)

        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=validation_steps,
            epochs=1,
            callbacks=[early_stopping, model_checkpoint, reduce_lr]
        )
        
        current_epoch += 1
        best_val_accuracy = max(best_val_accuracy, max(history.history['val_accuracy']))
        
        print(f"Epoch {current_epoch}/{max_epochs} - Best validation accuracy: {best_val_accuracy:.4f}")

        if best_val_accuracy >= target_accuracy:
            print(f"Target accuracy of {target_accuracy*100:.2f}% reached. Stopping training.")
            break

    # Save the entire model for later retraining
    model.save('emotion_recognition_model_custom_final.keras')
    # Save the model architecture
    with open('emotion_recognition_architecture_custom_final.json', 'w') as f:
        f.write(model.to_json())

    # Save the model weights
    model.save_weights('emotion_recognition_model_custom_weights_final.h5')
    
    return history

def main():

#     # Load training and test data
#     train_images, train_labels = load_images_from_folder(os.path.join(data_path, 'train'))
#     test_images, test_labels = load_images_from_folder(os.path.join(data_path, 'test'))

#     train_images = np.array(train_images)
#     test_images = np.array(test_images)

#     # One-hot encode the labels
#     lb = LabelBinarizer()
#     train_labels = lb.fit_transform(train_labels)
#     test_labels = lb.transform(test_labels)

#     # Data Augmentation
#     train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         rotation_range=20,
#         zoom_range=0.2,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         horizontal_flip=True
#     )

#     val_datagen = ImageDataGenerator(rescale=1./255)

#     train_generator = train_datagen.flow(train_images, train_labels, batch_size=64, shuffle=True)
#     val_generator = val_datagen.flow(test_images, test_labels, batch_size=64, shuffle=False)

#     # Create the model
#     #model = create_enhanced_custom_model()
#     # Load the model architecture from JSON file
#     with open('emotion_recognition_architecture_custom_final.json', 'r') as json_file:
#         model_json = json_file.read()

#     model = model_from_json(model_json)

# # Load the weights from HDF5 file
#     model.load_weights('emotion_recognition_model_custom_weights_final.h5')
    
#     # Alternatively, if you want to load an existing model, uncomment the following line
#     # model = load_model('best_model_custom_epoch_0.6094.keras')

#     # Compile the model
#     model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

#     history = train_model(model, train_generator, val_generator, target_accuracy=0.90)

    # # Plotting accuracy and loss
    # plt.figure(figsize=(12, 4))

    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['accuracy'], label='train_accuracy')
    # plt.plot(history.history['val_accuracy'], label='val_accuracy')
    # plt.title('Model Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['loss'], label='train_loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    # plt.title('Model Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.show()




    # Load the model architecture from JSON file
    with open('emotion_recognition_architecture_custom_final.json', 'r') as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json)

    # Load the weights from HDF5 file
    model.load_weights('emotion_recognition_model_custom_weights_final.h5')

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Load test data
    test_images, test_labels = load_images_from_folder(os.path.join(data_path, 'test'))
    test_images = np.array(test_images)
    lb = LabelBinarizer()
    test_labels = lb.fit_transform(test_labels)

    # # Normalize and wrap in generator
    # test_datagen = ImageDataGenerator(rescale=1./255)
    # val_generator = test_datagen.flow(test_images, test_labels, batch_size=64, shuffle=False)

    # # Evaluate
    # val_loss, val_accuracy = model.evaluate(val_generator)
    # print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    # print(f"Validation Loss: {val_loss:.4f}")

   
    # from sklearn.metrics import confusion_matrix, classification_report
    # import seaborn as sns

    # #  Predict class probabilities
    # pred_probs = model.predict(val_generator)
    # y_pred = np.argmax(pred_probs, axis=1)
    # y_true = np.argmax(test_labels, axis=1)  # Assuming one-hot encoding

    # #  Compute confusion matrix
    # cm = confusion_matrix(y_true, y_pred)
    # report = classification_report(y_true, y_pred, target_names=emotions)

    # #: Display confusion matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt='d', xticklabels=emotions, yticklabels=emotions, cmap='Blues')
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.title('Confusion Matrix')
    # plt.show()

    # # Step 4: Print classification report
    # print("Classification Report:")
    # print(report)
    


    # Test the model with an image from the test folder
    test_image_path = 'data/test/angry/PrivateTest_41546093.jpg'  # Replace with your test image path

    # Display the image
    display_image(test_image_path)

    # Predict the emotion
    predicted_emotion = predict_emotion(test_image_path, model)
    print(f'The predicted emotion for the image is: {predicted_emotion}')

if __name__ == "__main__":
    main()
