import cv2
import dlib
import numpy as np

# Load the pre-trained facial landmark detection model
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Replace the IP address and port number with your phone's IP address and port number
url = 'http://120.120.121.181:8080/video'
cap = cv2.VideoCapture(0)

# Define the desired width and height
width = 640
height = 480

while True:

    ret, img = cap.read()
    if not ret:
        print('Error fetching frame')
        break

    # Resize the frame to the desired width and height
    # img = cv2.resize(img, (width, height))

    # Detect faces in the input image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray, 0)

    # Loop over the detected faces
    for face in faces:

        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        # Get the facial landmarks
        landmarks = predictor(gray, face)
        # for n in range(0, 68):
        #     x = landmarks.part(n).x
        #     y = landmarks.part(n).y
        #     cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        #     cv2.putText(img, str(n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Extract the eye landmarks and draw circles on them
        left_eye_1 = landmarks.part(36)
        left_eye_2 = landmarks.part(39)
        right_eye_1 = landmarks.part(42)
        right_eye_2 = landmarks.part(45)

        nose_midpoint = (
            (landmarks.part(31).x + landmarks.part(35).x) // 2, (landmarks.part(31).y + landmarks.part(35).y) // 2)
        chin_midpoint = (
            (landmarks.part(7).x + landmarks.part(9).x) // 2, (landmarks.part(7).y + landmarks.part(9).y) // 2)

        left_eye_center = ((left_eye_1.x + left_eye_2.x) // 2, (left_eye_1.y + left_eye_2.y) // 2)
        right_eye_center = ((right_eye_1.x + right_eye_2.x) // 2, (right_eye_1.y + right_eye_2.y) // 2)
        mid_point = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)

        # distance b/w eyes
        distance = np.sqrt(
            (left_eye_center[0] - right_eye_center[0]) ** 2 + (left_eye_center[1] - right_eye_center[1]) ** 2)

        if chin_midpoint[0] == nose_midpoint[0]:
            x1, y1 = nose_midpoint[0], y
            x2, y2 = nose_midpoint[0], y + h
            a1, b1 = nose_midpoint[0], y
            a2, b2 = nose_midpoint[0], y + h
        else:
            slope = (chin_midpoint[1] - nose_midpoint[1]) / (chin_midpoint[0] - nose_midpoint[0])
            c = nose_midpoint[1] - slope * nose_midpoint[0]
            left_c = left_eye_center[1] - slope * left_eye_center[0]
            right_c = right_eye_center[1] - slope * right_eye_center[0]

            x1, y1 = (y - c) / slope, y
            x2, y2 = (y + h - c) / slope, y + h

            left_new_y = left_eye_center[1] - int(0.6 * distance)
            right_new_y = right_eye_center[1] - int(0.6 * distance)

            dot1_left_x, dot1_left_y = (left_new_y - left_c) / slope, left_new_y
            # dot2_left_x, dot2_left_y = (y + h - left_c) / slope, y + h

            dot1_right_x, dot1_right_y = (right_new_y - right_c) / slope, right_new_y
            # dot2_right_x,dot2_right_y = (y + h - right_c) / slope, y + h

        # STATIC X DOT and DYNAMIC Y DISTANCE
        # dot1_pos = (left_eye_center[0], left_eye_center[1] - int(0.6 * distance))
        # dot2_pos = (right_eye_center[0], right_eye_center[1] - int(0.6 * distance))
        # dot3_pos = (mid_point[0], mid_point[1] - int(0.6 * distance))

        # DYNAMIC X AND Y POINT
        dot1_pos = (int(dot1_left_x), dot1_left_y)
        dot2_pos = (int(dot1_right_x), dot1_right_y)

        cv2.line(img, (round(x1), y1), (round(x2), y2), (0, 255, 0), 2)

        # cv2.line(img, (round(dot1_left_x), dot1_left_y), (round(dot2_left_x), dot2_left_y), (0, 255, 0), 2)
        # cv2.line(img, (round(dot1_right_x), dot1_right_y), (round(dot2_right_x), dot2_right_y), (0, 255, 0), 2)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.circle(img, dot1_pos, 2, (255, 0, 0), -1)
        cv2.circle(img, dot2_pos, 2, (255, 0, 0), -1)
        # cv2.circle(img, dot3_pos, 2, (0, 255, 0), -1)

        # cv2.circle(img, left_eye_center, 2, (0, 255, 0), -1)
        # cv2.circle(img, right_eye_center, 2, (0, 255, 0), -1)
        # cv2.circle(img, mid_point, 2, (0, 0, 0), -1)

        cv2.circle(img, nose_midpoint, 2, (0, 255, 0), -1)
        cv2.circle(img, chin_midpoint, 2, (0, 255, 0), -1)

    # Show the output image
    cv2.imshow('output', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
