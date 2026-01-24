import cv2
import sys
import numpy as np
import os

#Using Haar Cascade file, pre-trained face detection model
HaarFile = "C:\OpenCV with python\L8\DataSets\haarcascade_frontalface_default.xml"
size = 4
print("Recognizing face, please be in sufficient light.")

(images,labels,names,id) = ([],[],{},0) #images to store face images, labels to store numeric ids, names for the persons name,id is a unique number for each person

Dataset = "C:\OpenCV with python\L8\DataSets"
sub_dataset = "C:\OpenCV with python\L8\DataSets\Chris"
path = os.path.join(Dataset,sub_dataset)

for (sub_dataset, Dataset, files) in os.walk(Dataset):
    for sub_dataset in Dataset:
        names[id] = sub_dataset
        subject_path = os.path.join(Dataset,sub_dataset)
        for file_name in os.listdir(subject_path):
            path = subject_path + "/" + file_name
            label = id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
    id += 1

(images,labels) = [np.array(lis)
                   for lis in [images,labels]]

#Creating a face recognizer
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images,labels)


(width, height) = (130,100)
FaceCascade = cv2.CascadeClassifier(HaarFile)
webcam = cv2.VideoCapture(0) #0 for the laptop cam and 1 for external cam

count = 1
while 1:
    (_,im) = webcam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    face = FaceCascade.detectMultiScale(1.3,5)
    for (x,y,w,h) in face:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,255),2)
        face = gray[y:y+h,x:x+w]
        face_resize = cv2.resize(face,(width,height))

        prediction = model.predict(face_resize)

        cv2.imshow("Face",im)
        key = cv2.waitKey(10)
        if key == 27:
            break