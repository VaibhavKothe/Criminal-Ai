import cv2

video_cap = cv2.VideoCapture(0)
while True:
    ret, img = video_cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    cv2.imshow("video_live", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video_cap.release()
cv2.destroyAllWindows()