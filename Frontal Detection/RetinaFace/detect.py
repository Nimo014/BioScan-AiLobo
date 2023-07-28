import cv2
import numpy as np
from retinaface import RetinaFace
import matplotlib.pyplot as plt

# Open the video stream
cap = cv2.VideoCapture(0)

# Initialize RetinaFace detector with minimal configuration

# Loop over the detected faces
def int_tuple(t):
    return tuple(int(x) for x in t)

while True:
    # Read a frame from the video stream
    ret, img = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    faces = RetinaFace.detect_faces(img)
    for key in faces:
        identity = faces[key]

        # ---------------------
        confidence = identity["score"]

        rectangle_color = (255, 255, 255)

        landmarks = identity["landmarks"]
        diameter = 1
        cv2.circle(img, int_tuple(landmarks["left_eye"]), diameter, (0, 0, 255), -1)
        cv2.circle(img, int_tuple(landmarks["right_eye"]), diameter, (0, 0, 255), -1)
        cv2.circle(img, int_tuple(landmarks["nose"]), diameter, (0, 0, 255), -1)
        cv2.circle(img, int_tuple(landmarks["mouth_left"]), diameter, (0, 0, 255), -1)
        cv2.circle(img, int_tuple(landmarks["mouth_right"]), diameter, (0, 0, 255), -1)

        facial_area = identity["facial_area"]

        cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), rectangle_color, 1)

        plt.imshow(img[:, :, ::-1])
        plt.axis('off')

plt.show()

# Release resources
cap.release()
cv2.destroyAllWindows()
