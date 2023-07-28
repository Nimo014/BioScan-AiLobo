import cv2
import numpy as np
from scipy.spatial import distance

# Load the pre-trained Caffe model for face detection
configFile = "models/deploy.prototxt"
modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Load the camera and start capturing video frames
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    if not ret:
        break

    # Detect faces in the input image
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()

    # Loop over the detected faces
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            # Get the coordinates of the face bounding box
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Extract the face region from the input image
            face = img[y1:y2, x1:x2]

            # Detect eyes in the face region
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            cv2.rectangle(img, (x1, y1), (x2, y2), (205, 92, 92), 2)
            # Calculate the distance between the eyes and the midpoint between them
            if len(eyes) == 2:
                (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes
                eye1_center = (int(x1 + ex1 + ew1 / 2), int(y1 + ey1 + eh1 / 2))
                eye2_center = (int(x1 + ex2 + ew2 / 2), int(y1 + ey2 + eh2 / 2))
                # eye_distance = distance.euclidean(eye1_center, eye2_center)
                # eye_midpoint = ((eye1_center[0] + eye2_center[0]) // 2, (eye1_center[1] + eye2_center[1]) // 2)
                #
                # # Calculate the positions of the dots 2cm above the eyes and equidistant from each other from the middle
                # dot_distance = eye_distance / 2
                # dot_offset = int(np.sqrt((2 * dot_distance) ** 2 - eye_distance ** 2 / 4))
                # dot1_position = (eye_midpoint[0] - dot_offset, eye_midpoint[1] - int(0.02 * eye_distance))
                # dot2_position = (eye_midpoint[0] + dot_offset, eye_midpoint[1] - int(0.02 * eye_distance))

                # Draw the dots on the input image
                cv2.circle(img, eye1_center, 5, (0, 255, 0), -1)
                cv2.circle(img, eye2_center, 5, (0, 255, 0), -1)

    cv2.imshow('Real-time face detection', img) # display the image
    if cv2.waitKey(1) == ord('q'): # press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
