import cv2
import numpy as np

# Load the Haar Cascade classifier for the right ear
right_ear_cascade = cv2.CascadeClassifier('./cascade_lateral_ears_opencv.xml')

if right_ear_cascade.empty():
    raise IOError('Unable to load the right ear cascade classifier xml file')

url = 'http://192.168.1.148:8080/video'
cap = cv2.VideoCapture(0)
BRIGHTNESS_LEVEL = 100
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_norm = clahe.apply(gray)

    right_ear = right_ear_cascade.detectMultiScale(gray_norm, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in right_ear:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Plot dots on the right ear
        cv2.circle(frame, (x, int(y+h*0.25)), 4, (0, 255, 0), 1)
        cv2.circle(frame, (int(x-w*0.3), int(y + h * 0.25)), 4, (0, 255, 0), 1)

    cv2.imshow('Left Ear Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
