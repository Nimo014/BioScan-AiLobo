import cv2

# Load the Haar Cascade classifier for the left ear
import numpy as np

left_ear_cascade = cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')
# Load the Haar Cascade classifier for the right ear
right_ear_cascade = cv2.CascadeClassifier('haarcascade_mcs_rightear.xml')

if left_ear_cascade.empty():
    raise IOError('Unable to load the left ear cascade classifier xml file')

if right_ear_cascade.empty():
    raise IOError('Unable to load the right ear cascade classifier xml file')

cap = cv2.VideoCapture(0)
scaling_factor = 0.8
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    left_ear = left_ear_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=3)
    right_ear = right_ear_cascade.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=3)

    for (x, y, w, h) in left_ear:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

    for (x, y, w, h) in right_ear:
        print(right_ear)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

    scale_factor = 0.1
    # PLOT DOTS ON EARS
    for (x, y, w, h) in left_ear:
        distance = np.sqrt(((x + w) - y) ** 2 + ((x + y) - (y + h)) ** 2)
        cv2.circle(frame, (x + w, int(y + y * scale_factor)), 4, (0, 255, 0), 1)

    for (x, y, w, h) in right_ear:
        distance = np.sqrt((x - y) ** 2 + (x - (y + h)) ** 2)
        cv2.circle(frame, (x, int(y + y * scale_factor)), 4, (0, 255, 0), 1)

    cv2.imshow('Ear Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
