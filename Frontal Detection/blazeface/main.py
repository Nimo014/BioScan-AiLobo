# IMPORTING LIBRARIES
import math
import time

import cv2
import mediapipe as mp
import numpy as np

# INITIALIZING OBJECTS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
url = 'http://192.168.1.148:4747/video'
cap = cv2.VideoCapture(0)

fps_start_time = time.time()
fps_counter = 0
fps_interval = 1  # Calculate the FPS over 1 second intervals
fps = 0
# DETECT THE FACE LANDMARKS
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:

        #success, image = cap.read()
        image = cv2.imread("./test_1.jpeg", cv2.IMREAD_COLOR)
        # Flip the image horizontally and convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Detect the face landmarks
        results = face_mesh.process(image)

        # To improve performance
        image.flags.writeable = True

        # Convert back to the BGR color space
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the face mesh annotations on the image.
        if results.multi_face_landmarks and len(results.multi_face_landmarks) == 1:

            for face_landmarks in results.multi_face_landmarks:
                left_eye_center = face_landmarks.landmark[159]
                right_eye_center = face_landmarks.landmark[386]

                # Draw circles at the left and right eye center coordinates
                for landmark in [left_eye_center, right_eye_center]:
                    x = landmark.x
                    y = landmark.y

                    shape = image.shape
                    relative_x = int(x * shape[1])
                    relative_y = int(y * shape[0])

                    cv2.circle(image, (relative_x, relative_y), radius=2, color=(225, 0, 100), thickness=1)

                # Compute the bounding box coordinates
                x_values = [lmk.x for lmk in face_landmarks.landmark]
                y_values = [lmk.y for lmk in face_landmarks.landmark]
                x1, y1 = int(min(x_values) * image.shape[1]), int(min(y_values) * image.shape[0])
                x2, y2 = int(max(x_values) * image.shape[1]), int(max(y_values) * image.shape[0])

                # Draw the bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                cv2.putText(image, 'Face Detection', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Compute the eye line and parallel line directions
                left_eye = (int(left_eye_center.x * shape[1]), int(left_eye_center.y * shape[0]))
                right_eye = (int(right_eye_center.x * shape[1]), int(right_eye_center.y * shape[0]))

                eye_direction = np.array(right_eye) - np.array(left_eye)
                eye_direction = eye_direction / np.linalg.norm(eye_direction)
                line_perp = np.array([-eye_direction[1], eye_direction[0]])


                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                line_width = 40  # adjust as desired

                # Draw the eye lines and parallel lines
                cv2.line(image, left_eye, right_eye, (255, 0, 0), thickness=2)
                # left_eye_offset = left_eye - line_width * line_perp
                # right_eye_offset = right_eye - line_width * line_perp
                # cv2.line(image, tuple(left_eye_offset.astype(int)), tuple(right_eye_offset.astype(int)),
                #          (0, 0, 255), thickness=2)
                #
                # # Draw a circle around the left eye position
                # cv2.circle(image, tuple(left_eye_offset.astype(int)), 2, (0, 255, 0), thickness=-1)
                #
                # # Draw a circle around the right eye position
                # cv2.circle(image, tuple(right_eye_offset.astype(int)), 2, (0, 255, 0), thickness=-1)


                # Find the midpoint of the line
                mid_point = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

                # Calculate the angle of the line
                angle = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])

                # Calculate the coordinates of the endpoints of the perpendicular line
                offset = 0.15 * distance
                left_dot_1_x = int(left_eye[0] + offset * math.cos(angle - math.pi / 2))
                left_dot_1_y = int(left_eye[1] + offset * math.sin(angle - math.pi / 2))

                right_dot_1_x = int(right_eye[0] + offset * math.cos(angle - math.pi / 2))
                right_dot_1_y = int(right_eye[1] + offset * math.sin(angle - math.pi / 2))

                cv2.circle(image, (left_dot_1_x, left_dot_1_y),radius=3, color=(0, 0, 0), thickness=1)
                cv2.circle(image, (right_dot_1_x, right_dot_1_y),radius=3, color=(0, 0, 0), thickness=1)

        else:
            cv2.putText(image, 'No Face Detection', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Update the FPS counter variables
        fps_counter += 1
        if (time.time() - fps_start_time) > fps_interval:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_counter = 0
        cv2.putText(image, f'FPS: {round(fps)}', (image.shape[1]-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        # Display the image
        cv2.imshow('MediaPipe FaceMesh', image)

        # Terminate the process
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()