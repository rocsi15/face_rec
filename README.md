# emotion_rec
# **TRAINING MODEL OF FACIAL EXPRESSION RECOGNITION WITH REAL TIME DETECTION** 
## **Versions**
***FER-2013 DATASET***\
python version: 3.11.4\
VS Code 1.98.2
## **Summary**
The goal of this study was building a deep learning model on a specific dataset, to implement and offer facial emotion recognition, which aims to reliably translate facial information into human emotions. ***The FER-2013 dataset was used for training*** in the first stage of the training model. In addition to properly predicting the emotional state of the tested image from the test file, the model is being trained with the objective to create a webcam function that uses a deep learning framework and can predict emotion in real-time, frame by frame.  As long as there are no shadows or obstructions like glasses or masks, the function will properly anticipate the emotion by analyzing each frame supplied by the video input.  It can even predict the presence of many people in the frame when all the conditions are ideal.  The model's validation accuracy demonstrated that it did rather well in classifying the seven emotion classes (happy, sad, neutral, fear, surprise, angry, and disgust).  According to the classification report and confusion matrix analysis, the model performs rather well for the most consistently detected emotions, "happy" and "neutral."  But because of the class difference and the visual similarity of some negative emotions, "disgust" and "anger" were very challenging for it. The training model's output had an adequate validation accuracy of 50.29%. 
## The process of implementation:
<img width="942" height="455" alt="image" src="https://github.com/user-attachments/assets/2b18eebc-6e6a-4cc7-8ffa-c4ba721c5fce" />\
**facerecon.py** webcam app\
**fer.py** training and testing images app\
**test_emotion** tests which images from the test file the app detects
