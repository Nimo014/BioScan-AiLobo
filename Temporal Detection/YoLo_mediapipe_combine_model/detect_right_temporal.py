# IMPORTING LIBRARIES
import math
import time
import numpy as np
from convert_annotations import denormalize
import os
import cv2
import torch

print("START")


class Detector:
    weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), '', 'best.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

    # By calling with image_name, the model resizes the picture as appropriate
    def detect(self, image_name, ear='right'):
        detections = []
        results = self.model(image_name, size=416)
        for tensor1, tensor2 in zip(results.xywhn, results.xywh):
            for result1, result2 in zip(tensor1, tensor2):
                x_norm, y_norm, w_norm, h_norm, _, _ = result1.numpy()
                x_, y_, w_, h_, _, _ = result2.numpy()
                x, y, w, h = denormalize(x_norm, y_norm, w_norm, h_norm)
                detections.append((x, y, w, h))

        return detections


print("END")

import mediapipe as mp

# INITIALIZING OBJECTS
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

fps_start_time = time.time()
fps_counter = 0
fps_interval = 1  # Calculate the FPS over 1 second intervals
fps = 0

def lineLineIntersection(A, B, C, D):
    # Line AB represented as a1x + b1y = c1

    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1 * (A[0]) + b1 * (A[1])

    # Line CD represented as a2x + b2y = c2
    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2 * (C[0]) + b2 * (C[1])

    determinant = a1 * b2 - a2 * b1
    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant

    return x, y

# Yolo model object
detector = Detector()

# DETECT THE FACE LANDMARKS
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        success, image = cap.read()

        # Flip the image horizontally and convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Detect the face landmarks
        results = face_mesh.process(image)

        # Detect Ear
        detected_loc = detector.detect(image)

        # To improve performance
        image.flags.writeable = True

        # Convert back to the BGR color space
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the face mesh annotations on the image.
        if results.multi_face_landmarks and len(results.multi_face_landmarks) == 1:
            face_landmarks = results.multi_face_landmarks[0]

            point_1 = face_landmarks.landmark[372]
            point_2 = face_landmarks.landmark[433]
            point_3 = face_landmarks.landmark[356]

            shape = image.shape
            point_1 = (point_1.x * shape[1], point_1.y * shape[0])
            point_2 = (point_2.x * shape[1], point_2.y * shape[0])
            point_3 = (point_3.x * shape[1], point_3.y * shape[0])

            # EAR Bounding Box
            for x, y, w, h in detected_loc:
                center_x = x + int(w / 2)
                center_y = y + int(h / 2)

                # VERTICAL JAW LINE
                dx = point_2[0] - point_1[0]
                dy = point_2[1] - point_1[1]
                slope_v = dy / dx
                intercept = point_1[1] - slope_v * point_1[0]
                cv2.line(image, (int(point_1[0]), int(point_1[1])),
                         (int(point_2[0]), int(point_2[1])), color=(0, 255, 0), thickness=2)

                # HORIZONTAL EYE LINE
                dx = point_3[0] - point_1[0]
                dy = point_3[1] - point_1[1]
                slope_h = dy / dx

                # Draw line between point 2 and point 3
                cv2.line(image, (int(point_1[0]), int(point_1[1])),
                         (int(point_3[0]), int(point_3[1])), color=(0, 255, 0), thickness=2)

                # LINE PASSING THROUGH EAR
                midpoint_x = int((point_1[0] + point_2[0]) / 2)
                midpoint_y = int((point_1[1] + point_2[1]) / 2)

                total_distance = abs(slope_v * (center_x - midpoint_x) - (center_y - midpoint_y)) / math.sqrt(
                    slope_v ** 2 + 1)
                dist = 0.8 * total_distance

                # Unit Vector
                dx = point_2[0] - point_1[0]
                dy = point_2[1] - point_1[1]
                line_length = np.sqrt(dx ** 2 + dy ** 2)
                dx_unit = dx / line_length
                dy_unit = dy / line_length
                p_dx = -dy_unit
                p_dy = dx_unit

                point_1_parallel = (int(point_1[0] - p_dx * dist), int(point_1[1] - p_dy * dist))
                point_2_parallel = (int(point_2[0] - p_dx * dist), int(point_2[1] - p_dy * dist))

                x, y = lineLineIntersection(point_1, point_3, point_1_parallel, point_2_parallel)
                cv2.circle(image, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=3)
                cv2.line(image, point_1_parallel, point_2_parallel, (0, 0, 255), 2)

        cv2.putText(image, 'Face Detection', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the FPS on the image
        fps_counter += 1
        if (time.time() - fps_start_time) > fps_interval:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()

        cv2.putText(image, 'FPS: {:.2f}'.format(fps), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Mesh', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
