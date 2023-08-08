import cv2
import numpy as np
import face_recognition


imgEllon = face_recognition.load_image_file('images_face_rec/elon musk.png')
imgEllon = cv2.cvtColor(imgEllon,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('images_face_rec/elon musk test.png')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

cv2.imshow('Ellon Musk',imgEllon )
cv2.imshow('Ellon Test',imgTest)
cv2.waitKey(0)