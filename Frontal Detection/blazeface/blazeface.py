# IMPORTING LIBRARIES
import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import math
import time
import numpy as np

# INITIALIZING OBJECTS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

url = 'http://192.168.1.52:4747/video'
cap = cv2.VideoCapture(0)

fps_start_time = time.time()
fps_counter = 0
fps_interval = 1  # Calculate the FPS over 1 second intervals
fps = 0



# DETECT THE FACE LANDMARKS
with mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.3) as face_mesh:
    while True:
        success, image = cap.read()

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
            x_min, y_min, x_max, y_max = np.inf, np.inf, 0, 0
            for face_landmarks in results.multi_face_landmarks:
                shape = image.shape
                for landmark in face_landmarks.landmark:
                    x, y = landmark.x * image.shape[1], landmark.y * image.shape[0]
                    x_min, y_min = min(x, x_min), min(y, y_min)
                    x_max, y_max = max(x, x_max), max(y, y_max)
                face_bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

                # Draw the face bounding box
                cv2.rectangle(image, face_bbox, color=(0, 255, 0), thickness=2)

                left_frontal_landmarks = list([face_landmarks.landmark[69], face_landmarks.landmark[108]])
                right_frontal_landmarks = list([face_landmarks.landmark[337],face_landmarks.landmark[299]])
                left_eye_center = face_landmarks.landmark[159]
                right_eye_center = face_landmarks.landmark[386]

                left_eye = (int(left_eye_center.x * shape[1]), int(left_eye_center.y * shape[0]))
                right_eye = (int(right_eye_center.x * shape[1]), int(right_eye_center.y * shape[0]))

                for landmark in [left_frontal_landmarks[0],right_frontal_landmarks[1]]:
                    x = landmark.x
                    y = landmark.y

                    # Convert the landmark coordinates to pixel values
                    relative_x = int(x * shape[1])
                    relative_y = int(y * shape[0])

                    # Draw a circle at the landmark coordinates
                    cv2.circle(image, (relative_x, relative_y), radius=3, color=(0, 0, 0), thickness=1)

                # Calculate midpoint between left frontal landmarks 69 and 108
                left_midpoint_x = (left_frontal_landmarks[0].x + left_frontal_landmarks[1].x) / 2
                left_midpoint_y = (left_frontal_landmarks[0].y + left_frontal_landmarks[1].y) / 2
                left_midpoint = (int(left_midpoint_x * shape[1]), int(left_midpoint_y * shape[0]))

                # Draw circle at the midpoint of left frontal landmarks
                cv2.circle(image, left_midpoint, radius=3, color=(0, 0, 0), thickness=1)

                # Calculate midpoint between right frontal landmarks 299 and 337
                right_midpoint_x = (right_frontal_landmarks[0].x + right_frontal_landmarks[1].x) / 2
                right_midpoint_y = (right_frontal_landmarks[0].y + right_frontal_landmarks[1].y) / 2
                right_midpoint = (int(right_midpoint_x * shape[1]), int(right_midpoint_y * shape[0]))

                # Draw circle at the midpoint of right frontal landmarks
                cv2.circle(image, right_midpoint, radius=3, color=(0, 0, 0), thickness=1)
                # Calculate midpoint between left frontal landmarks 69 and 108
                left_midpoint_x = (left_frontal_landmarks[0].x + left_frontal_landmarks[1].x) / 2
                left_midpoint_y = (left_frontal_landmarks[0].y + left_frontal_landmarks[1].y) / 2
                left_midpoint = (int(left_midpoint_x * shape[1]), int(left_midpoint_y * shape[0]))

                # Draw circle at the midpoint of left frontal landmarks
                cv2.circle(image, left_midpoint, radius=3, color=(0, 0, 0), thickness=1)

                # Calculate midpoint between right frontal landmarks 299 and 337
                right_midpoint_x = (right_frontal_landmarks[0].x + right_frontal_landmarks[1].x) / 2
                right_midpoint_y = (right_frontal_landmarks[0].y + right_frontal_landmarks[1].y) / 2
                right_midpoint = (int(right_midpoint_x * shape[1]), int(right_midpoint_y * shape[0]))

                # Draw circle at the midpoint of right frontal landmarks
                cv2.circle(image, right_midpoint, radius=3, color=(0, 0, 0), thickness=1)

                mid_point = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

                # Calculate the angle of the line
                angle = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])

                distance = math.sqrt((y_max-y_min)**2 + (x_max-x_min)**2)
                # Calculate the coordinates of the endpoints of the perpendicular line
                offset = 0.3 * distance
                x1 = int(mid_point[0] + offset * math.cos(angle + math.pi / 2))
                y1 = int(mid_point[1] + offset * math.sin(angle + math.pi / 2))

                x2 = int(mid_point[0] + offset * math.cos(angle - math.pi / 2))
                y2 = int(mid_point[1] + offset * math.sin(angle - math.pi / 2))

                cv2.line(image, (x2,y2),(x1,y1), color=(200, 0, 100), thickness=2)
                cv2.putText(image, 'Face Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, 'No Face Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
