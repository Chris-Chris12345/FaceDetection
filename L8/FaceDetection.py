import cv2
import sys
import numpy as np
import os

#Using Haar Cascade file, pre-trained face detection model
HaarFile = "C:\OpenCV with python\L8\DataSets\haarcascade_frontalface_default.xml"

Dataset = "C:\OpenCV with python\L8\DataSets"
sub_dataset = "C:\OpenCV with python\L8\DataSets\Chris"
path = os.path.join(Dataset,sub_dataset)

if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (130,100)
FaceCascade = cv2.CascadeClassifier(HaarFile)
webcam = cv2.VideoCapture(0) #0 for the laptop cam and 1 for external cam

count = 1
while count < 30:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    face = FaceCascade.detectMultiScale(gray,1.3,4) #1.3 is the scale factor for image size reduction, 4 are the minimum neighbor (higher = more accuracy) it helps you in getting x,w,h
    
    for (x,y,w,h) in face:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y+h,x:x+w]
        face_resize = cv2.resize(face,(width,height))

        cv2.imwrite('% s/% s.png'%(path,count),face_resize)
        count += 1

        cv2.imshow("Face",im)
        key = cv2.waitKey(10)
        if key == 27:
            break