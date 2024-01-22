# FER
Facial Emotion Recognition

**Project description** :
  This project detects up to multiple faces , drawing a bounding box around each and determines the emotion of each face deteceted.
  
  * Face detection : using haarcascade pretrained model ; link to model -> https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
  
  * Emotion classification : We trained a ConvNN on a dataset ; link to dataset -> https://www.kaggle.com/datasets/msambare/fer2013.
  
  * The application runs in a real time manner.

**Manual** :
  - For inference :
    ``` shell
    python EmoRec.py
    ```
  - For training check the notebook main.py

**Test Video* :
  - https://youtu.be/JYPJorg5VAU
  
