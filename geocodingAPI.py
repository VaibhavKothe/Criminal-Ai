import requests
import json

# Get the API key from the environment variable
api_key = os.environ['GEOCODING_API_KEY']

# Make the geocoding request
url = 'https://api.openstreetmap.org/api/geocode/json?lat={lat}&lon={lon}&apikey={api_key}'.format(
    lat=latitude,
    lon=longitude,
    api_key=api_key
)

response = requests.get(url)

# Parse the response JSON
data = json.loads(response.content)

# Get the latitude and longitude
latitude = data['results'][0]['geometry']['coordinates'][1]
longitude = data['results'][0]['geometry']['coordinates'][0]

# Print the latitude and longitude
print(latitude, longitude)







import face_recognition
import googlemaps

# Get the API key for the Google Maps Geocoding API
API_KEY = "YOUR_API_KEY"

# Create a Google Maps client
gmaps = googlemaps.Client(key=API_KEY)

# Create a face recognition object
face_recognizer = face_recognition.FaceRecognizer()

# Start a live video stream
video_stream = cv2.VideoCapture(0)

while True:
    # Grab a frame from the video stream
    ret, frame = video_stream.read()

    # Detect faces in the frame
    faces = face_recognition.face_locations(frame)

    # Recognize the faces in the frame
    for face_location in faces:
        face_encoding = face_recognition.face_encodings(frame, face_location)[0]
        name = face_recognition.face_identify(face_encoding)

        # If the face is recognized, get the geo coordinates of the face
        if name:
            geo_coordinates = gmaps.geocode(name)[0]["geometry"]["location"]
            latitude = geo_coordinates["lat"]
            longitude = geo_coordinates["lng"]

            # Print the geo coordinates of the face
            print(f"The geo coordinates of {name} are {latitude}, {longitude}")

    # Show the frame
    cv2.imshow("Frame", frame)

    # Check if the user wants to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Stop the video stream
video_stream.release()

# Close all windows
cv2.destroyAllWindows()
