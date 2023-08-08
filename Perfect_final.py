import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import threading
import winsound  # Add this import for playing the buzzer sound
import pygame  # Add this import for playing the alert alarm sound

# Initialize Pygame
pygame.init()
# Load the alert alarm sound file
alarm_sound_file = "alert_alarm.wav"
pygame.mixer.init()
alert_alarm_sound = pygame.mixer.Sound(alarm_sound_file)

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

            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

        # Play the buzzer sound only if the name is not 'Unknown'
        if name != 'People':
            alert_alarm_sound.play()


encodeListKnown = findEncodings(photos)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

consecutive_frames = 0
consecutive_frames_threshold = 3
last_detected_name = ""


def face_recognition_thread():
    global last_detected_name
    global consecutive_frames

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
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
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(img, "Criminal", (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

                    if name == last_detected_name:
                        consecutive_frames += 1
                        if consecutive_frames >= consecutive_frames_threshold:
                            markAttendence(name)
                            consecutive_frames = 0
                    else:
                        last_detected_name = name
                        consecutive_frames = 1
                else:
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, "People", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        else:
            last_detected_name = ""
            consecutive_frames = 0

        cv2.imshow('Webcam', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
