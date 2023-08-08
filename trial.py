import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from geopy.geocoders import Nominatim
import pygame  # Add this import for playing the alert alarm sound

def get_geo_coordinates(location_name):
    geolocator = Nominatim(user_agent="geo_coordinates_app")
    location = geolocator.geocode(location_name)
    return location.latitude, location.longitude

def detect_face_and_get_coordinates():
    # Replace "0" with the appropriate camera index if you have multiple cameras.
    cap = cv2.VideoCapture(0)

    # Load a face detection model from OpenCV (Haar Cascade or DNN based models).
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    location_name = "Nagpur, India"  # Replace with your current location.

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for face detection.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame.
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw a rectangle around the face.
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Get the ROI (region of interest) containing the face.
            face_roi = frame[y:y+h, x:x+w]

            # Get the geo coordinates of the location.
            latitude, longitude = get_geo_coordinates(location_name)
            print(f"Face detected! Latitude: {latitude}, Longitude: {longitude}")

        # Display the video frame with face detection.
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_face_and_get_coordinates()


# Initialize Pygame
pygame.init()
# Load the alert alarm sound file
alarm_sound_file ="alert_alarm.wav"
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

# from PIL import Image

def markAttendence(name, img):
    with open('Attendence.csv', 'a+') as f:  # Open the file in append mode ('a+')
        f.seek(0)  # Move the file pointer to the beginning
        myDtaList = f.readlines()
        nameList = []
        print(myDtaList)

        for line in myDtaList:
            entry = line.split(',')
            nameList.append(entry[0])

        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')


        f.writelines(f'\n{name},{dtString},{latitude},{longitude}')



encodeListKnown = findEncodings(photos)
print('Encoding Complete')

# cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrames = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrames, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = photosNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, "Criminal", (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            # Play the buzzer sound only if the name is not 'People'

            alert_alarm_sound.play()

        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "People", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
