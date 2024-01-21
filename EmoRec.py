import cv2
from CNN_model import CNNModel
import torch
import torchvision.transforms as transforms


import numpy as np

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# VideoCapture(0 => webcam) otherwise provide the path of the saved video
cap = cv2.VideoCapture(0)

model = CNNModel()
checkpoint = torch.load("emoji_model2.pth")
model.load_state_dict(checkpoint["state_dict"])
model = model.to('cpu')

# front face detector (pretrained model) 
Classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # ret is a bool , True if the frame reading was successful
    # frame is the picture
    ret, frame = cap.read()

    # if ret is False
    if not ret:
        # closes video
        cap.release()
        cv2.destroyAllWindows()
        break

    
    # convert frame to gray scale to match the training set
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    
    # detecting the frontal faces for the current frame
    bounding_boxes = Classifier.detectMultiScale(frame,scaleFactor=1.3, minNeighbors=5)
    # whenever a face is detected, calssifier determines its bounding box
    # each bounding box = [x,y,w,h] => 
    # x,y : the upper right corner coordinates of the box
    # w,h : the width and the height of the box


    # for each bounding box
    for (x, y, w, h) in bounding_boxes:
        # drawing the boundig box onto the video frame (the original one not the gray one)
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        # rectangle takes the upper right corner : (x, y-50) and the lower left corner : (x+w, y+h+10) of the rectangle

        # cropping the image around the bounding box of the face to input only the face to the emotion detection model
        roi_gray_frame = gray_frame[y:y + h, x:x + w]

        
        # resize the cropped image to (48,48) like the training set
        cropped_img = cv2.resize(roi_gray_frame, (48, 48))
        # cv2.resize(roi_gray_frame, (48, 48)).shape = (48,48)
        # np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), 0).shape = (1,48,48)


        
        # predict
        emotion_prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))

        # write the emotion predicted on the video frame
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # show the frame after editting
    cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
    # to quit video press q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break