import time

import cv2
import dlib
import os
import math

# Get Current Dir path
path = os.path.dirname(__file__)

# Load the detector
detector = dlib.get_frontal_face_detector()

# shape_predictor_5_face_landmarks.dat
# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start the video capture
cap = cv2.VideoCapture(0)

# Minimum face size threshold (in pixels)
min_face_size = 400
frame = cv2.imread('C:\\Users\\nirav\\Documents\\BioScan\\Frontal_Detection\\images\\test_img.jpg')

# Convert the frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale frame
faces = detector(gray)

# When no face detected
if len(faces) == 0:
    cv2.putText(frame, 'No face detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# When more than one face detected
elif len(faces) > 1:
    cv2.putText(frame, 'Multiple face detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Only one face detected
else:
    face = faces[0]

    x, y, w, h = face.left(), face.top(), face.width(), face.height()

    landmarks = predictor(gray, face)

    for n in range(0, 68):
        a = landmarks.part(n).x
        b = landmarks.part(n).y
        cv2.circle(frame, (a, b), 2, (0, 255, 0), -1)
        cv2.putText(frame, str(n), (a, b), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # calculate the angle between the eyes
    eye_angle = math.atan2(landmarks.part(45).y - landmarks.part(36).y, landmarks.part(45).x - landmarks.part(36).x)
    left_eye_1 = landmarks.part(36)
    left_eye_2 = landmarks.part(39)
    right_eye_1 = landmarks.part(42)
    right_eye_2 = landmarks.part(45)

    nose_midpoint = (
        (landmarks.part(31).x + landmarks.part(35).x) // 2, (landmarks.part(31).y + landmarks.part(35).y) // 2)
    chin_midpoint = (
        (landmarks.part(7).x + landmarks.part(9).x) // 2, (landmarks.part(7).y + landmarks.part(9).y) // 2)

    # calculate the position of the dots above the eyes
    # left_eye_dot = (
    #     landmarks.part(37).x - int(0.05 * math.cos(eye_angle)), landmarks.part(37).y - int(0.05 * math.sin(eye_angle)))
    # right_eye_dot = (
    #     landmarks.part(44).x - int(0.05 * math.cos(eye_angle)), landmarks.part(44).y - int(0.05 * math.sin(eye_angle)))

    left_eye_center = ((left_eye_1.x + left_eye_2.x) // 2, (left_eye_1.y + left_eye_2.y) // 2)
    right_eye_center = ((right_eye_1.x + right_eye_2.x) // 2, (right_eye_1.y + right_eye_2.y) // 2)

    # calculate the position of the dot equidistant from the midpoint of the nose
    nose_dot = (nose_midpoint[0], nose_midpoint[1] - int(0.25 * math.sqrt(
        (landmarks.part(42).x - landmarks.part(39).x) ** 2 + (landmarks.part(42).y - landmarks.part(39).y) ** 2)))

    if chin_midpoint[0] == nose_midpoint[0]:
        x1, y1 = nose_midpoint[0], y
        x2, y2 = nose_midpoint[0], y+h
    else:
        slope = (chin_midpoint[1] - nose_midpoint[1]) / (chin_midpoint[0] - nose_midpoint[0])
        c = nose_midpoint[1] - slope * nose_midpoint[0]

        x1, y1 = (y-c)/slope, y
        x2, y2 = (y+h-c)/slope, y+h

    cv2.line(frame, (round(x1), y1), (round(x2), y2), (0, 0, 255), 3)


    # draw the dots on the image
    cv2.circle(frame, left_eye_center, 3, (0, 255, 0), -1)
    cv2.circle(frame, right_eye_center, 3, (0, 255, 0), -1)
    cv2.circle(frame, nose_dot, 3, (0, 255, 0), -1)
    cv2.circle(frame, nose_midpoint, 3, (0, 0, 0), -1)

    cv2.circle(frame, chin_midpoint, 3, (0, 0, 0), -1)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
# Display the resulting frame
cv2.imshow('Facial Landmarks', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
