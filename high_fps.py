import cv2
import numpy as np
import os
from datetime import datetime
import threading
import face_recognition

path = 'images_face_rec'
photos = []
photosNames = []
nameList = os.listdir(path)

for cls in nameList:
    currImg = cv2.imread(f'{path}/{cls}')
    photos.append(currImg)
    photosNames.append(os.path.splitext(cls)[0])

print(photosNames)

def findEncodings(photos):
    encodeList = []
    for img in photos:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendence(name):
    with open('Attendence.csv', 'r+') as f:
        myDtaList = f.readlines()
        nameList = []
        print(myDtaList)
        for line in myDtaList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(photos)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

consecutive_frames = 0
consecutive_frames_threshold = 3
last_detected_name = ""

# Load the pre-trained MobileNet SSD model for face detection
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
conf_threshold = 0.7

def detect_faces(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detections = net.forward()
    faces = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            faces.append((x1, y1, x2, y2))

    return faces

def face_recognition_thread():
    global last_detected_name
    global consecutive_frames

    while True:
        success, img = cap.read()
        if not success:
            break

        # Resize the image for faster processing
        imgS = cv2.resize(img, (0, 0), None, 0.5, 0.5)

        # Detect faces using the modified detect_faces function
        faceCurFrame = detect_faces(imgS)
        encodeCurFrames = face_recognition.face_encodings(imgS, faceCurFrame)

        if len(faceCurFrame) > 0:
            face_detected = True
        else:
            face_detected = False

        if face_detected:
            for encodeFace, faceLoc in zip(encodeCurFrames, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = photosNames[matchIndex].upper()
                    x1, y1, x2, y2 = faceLoc
                    cv2.rectangle(img, (x1*2, y1*2), (x2*2, y2*2), (0, 0, 255), 2)
                    cv2.rectangle(img, (x1*2, y2*2 - 35), (x2*2, y2*2), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, name, (x1*2 + 6, y2*2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                    if name == last_detected_name:
                        consecutive_frames += 1
                        if consecutive_frames >= consecutive_frames_threshold:
                            markAttendence(name)
                            consecutive_frames = 0
                    else:
                        last_detected_name = name
                        consecutive_frames = 1
                else:
                    x1, y1, x2, y2 = faceLoc
                    cv2.rectangle(img, (x1*2, y1*2), (x2*2, y2*2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1*2, y2*2 - 35), (x2*2, y2*2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, "Unknown", (x1*2 + 6, y2*2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    markAttendence("Unknown")

        else:
            last_detected_name = ""
            consecutive_frames = 0

        cv2.imshow('Webcam', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start the face recognition thread
face_recognition_thread()
