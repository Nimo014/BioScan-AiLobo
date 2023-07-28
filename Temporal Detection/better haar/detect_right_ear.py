import cv2
import numpy as np

# Load the Haar Cascade classifier for the left ear
left_ear_cascade = cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')

if left_ear_cascade.empty():
    raise IOError('Unable to load the left ear cascade classifier xml file')

url = 'http://192.168.1.148:8080/video'
cap = cv2.VideoCapture(url)
scaling_factor = 0.8
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_norm = clahe.apply(gray)

    left_ear = left_ear_cascade.detectMultiScale(gray_norm, scaleFactor=1.7, minNeighbors=4)

    for (x, y, w, h) in left_ear:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Plot dots on the left ear
        scale_factor = 0.15
        distance = np.sqrt(((x + w) - y) ** 2 + ((x + w) - (y + h)) ** 2)
        # cv2.circle(frame, (x + w, int(y + distance * scale_factor * (h) / distance)), 4, (0, 255, 0), 1)
        # cv2.circle(frame, (int(x + w + distance*0.1), int(y + distance * scale_factor * (h) / distance)), 4, (0, 255, 0), 1)

        cv2.circle(frame, (x+w, int(y + h * 0.25)), 4, (0, 255, 0), 1)
        cv2.circle(frame, (int((x+w) + w * 0.3), int(y + h * 0.25)), 4, (0, 255, 0), 1)

    cv2.imshow('Right Ear Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
