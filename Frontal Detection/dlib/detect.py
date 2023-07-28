import time

import cv2
import dlib
import os

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
while True:
    # Read the frame
    ret, frame = cap.read()

    if not ret:
        print("Could not read Image")
        exit()

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

        # if w * h >= 30000 and w * h <= 60000:
        if True:
            # # Getting Facial Landmark
            landmarks = predictor(gray, face)
            # for n in range(0, 68):
            #     x = landmarks.part(n).x
            #     y = landmarks.part(n).y
            #     cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            #     cv2.putText(frame, str(n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            left_eye_1 = landmarks.part(36)
            left_eye_2 = landmarks.part(39)
            right_eye_1 = landmarks.part(42)
            right_eye_2 = landmarks.part(45)
            nose_tip = landmarks.part(30)

            # Draw a vertical line passing through nose tip
            cv2.line(frame, (nose_tip.x, 0), (nose_tip.x, frame.shape[0]), (0, 255, 0), 1)

            cv2.putText(frame, 'face detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # # Getting coordinate eye and nose mid-point coordinate
            left_eye_center = ((left_eye_1.x + left_eye_2.x) // 2, (left_eye_1.y + left_eye_2.y) // 2)
            right_eye_center = ((right_eye_1.x + right_eye_2.x) // 2, (right_eye_1.y + right_eye_2.y) // 2)
            nose_center = (nose_tip.x, nose_tip.y)

            # Define dot offsets relative to eye centers
            dot_offset_x = 0.21  # Fraction of bounding box width
            dot_offset_y = 0.21  # Fraction of bounding box height

            # Getting left dot coordinates
            dot1_left = (nose_tip.x - int(w * dot_offset_x),
                         int(left_eye_center[1] - h * dot_offset_y))
            dot2_left = (nose_tip.x - int(w * dot_offset_x) - 5,
                         int(left_eye_center[1] - h * dot_offset_y))

            # Getting right dot coordinates
            dot1_right = (nose_tip.x + int(w * dot_offset_x),
                          int(right_eye_center[1] - h * dot_offset_y))
            dot2_right = (nose_tip.x + int(w * dot_offset_x) + 5,
                          int(right_eye_center[1] - h * dot_offset_y))

            # # Getting left dot coordinates
            # dot1_left = (nose_tip.x - 35, left_eye_center[1] - 45)
            # dot2_left = (nose_tip.x - 40, left_eye_center[1] - 45)
            #
            # # Getting right dot coordinates
            # dot1_right = (nose_tip.x + 35, right_eye_center[1] - 45)
            # dot2_right = (nose_tip.x + 40, right_eye_center[1] - 45)

            # Plotting left eye and left dot circle
            # cv2.circle(frame, left_eye_center, 2, (0, 255, 0), -1)
            cv2.circle(frame, dot1_left, 2, (0, 0, 255), -1)
            cv2.circle(frame, dot2_left, 2, (0, 0, 255), -1)

            # Plotting right eye and right dot circle
            # cv2.circle(frame, right_eye_center, 2, (0, 255, 0), -1)
            cv2.circle(frame, dot1_right, 2, (0, 0, 255), -1)
            cv2.circle(frame, dot2_right, 2, (0, 0, 255), -1)

            # Plotting Nose tip circle
            # cv2.circle(frame, nose_center, 2, (0, 255, 0), -1)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'Come Closer To Camera', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Facial Landmarks', frame)

    # Quit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()

# Destroy all windows
cv2.destroyAllWindows()
