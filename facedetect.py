from cv2 import cv2
from random import random

trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default1.xml')

# img = cv2.imread('pic1.jpg')
webcam = cv2.VideoCapture(0)

while True:
    sfr, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 10)

    cv2.imshow('Aryans face detector', frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
