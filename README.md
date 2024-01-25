# Facial Emotion Recognition (FER)

## Project Description

Facial Emotion Recognition (FER) is a project that detects multiple faces, draws bounding boxes around each, and determines the emotion of each detected face. The system uses a Haarcascade pre-trained model for face detection and a Convolutional Neural Network (ConvNN) trained on a dataset for emotion classification. The application operates in real-time.

### Face Detection
- Haarcascade Pre-trained Model: [Link to model](https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)

### Emotion Classification
- Convolutional Neural Network (ConvNN) trained on the FER2013 dataset: [Link to dataset](https://www.kaggle.com/datasets/msambare/fer2013)

## Manual

### For Inference
```shell
python EmoRec.py
```

### For Training
Check the notebook `main.py`

## Test Video
[Watch the demo video](https://youtu.be/JYPJorg5VAU)

## Installation

To install the required dependencies, use the following command:

```shell
pip install -r requirements.txt
```

## Requirements

- Python 3.x
- OpenCV
- TensorFlow
- Numpy
- ...

## Results

Include any relevant results, performance metrics, or accuracy scores from the FER system.

---

Feel free to customize the sections, add specific dependencies to the `requirements.txt` file, and include any other information that might be relevant for users and contributors.
