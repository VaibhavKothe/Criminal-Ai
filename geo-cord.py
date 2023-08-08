import cv2
import face_recognition
from geopy.geocoders import Nominatim

def get_geo_coordinates(location_name):
    geolocator = Nominatim(user_agent="geo_coordinates_app")
    location = geolocator.geocode(location_name)
    return location.latitude, location.longitude

def detect_face_and_get_coordinates():
    # Replace "0" with the appropriate camera index if you have multiple cameras.
    cap = cv2.VideoCapture(0)

    location_name = "Nagpur, India"  # Replace with your current location.

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame from BGR (OpenCV) to RGB (face_recognition).
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations in the frame.
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")

        for top, right, bottom, left in face_locations:
            # Draw a rectangle around the face.
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

            # Get the ROI (region of interest) containing the face.
            face_roi = frame[top:bottom, left:right]

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
