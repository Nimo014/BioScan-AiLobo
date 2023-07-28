import os
import cv2
import torch
from convert_annotations import denormalize
import math
import numpy as np


class Detector:
    weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), '', 'best.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

    # By calling with image_name, the model resizes the picture as appropriate
    def detect(self, image_name):
        detections = []
        results = self.model(image_name, size=416)

        for tensor1, tensor2 in zip(results.xywhn, results.xywh):
            for result1, result2 in zip(tensor1, tensor2):
                x_norm, y_norm, w_norm, h_norm, _, _ = result1.numpy()
                x_, y_, w_, h_, _, _ = result2.numpy()
                detections.extend([denormalize(x_norm, y_norm, w_norm, h_norm)])

        return detections


def plot_helix(frame, center_x, center_y, ear_width, ear_height):
    # Calculate the coordinates of the points on the helix of the ear
    radius_x = ear_width / 2
    radius_y = ear_height / 2
    num_points = 12
    angle_step = 2 * math.pi / num_points

    helix_points = []
    for i in range(num_points):
        angle = i * angle_step
        x = int(center_x + radius_x * math.cos(angle))
        y = int(center_y - radius_y * math.sin(angle))
        helix_points.append((x, y))

    # Plot the helix points on the frame
    for idx, point in enumerate(helix_points):
        cv2.circle(frame, point, 3, (0, 0, 255), -1)
        cv2.putText(frame, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def plot_edges(frame, center_x, center_y, ear_width, ear_height):
    # Extract the region of interest around the ear
    x1 = int(center_x - ear_width / 2)
    x2 = int(center_x + ear_width / 2)
    y1 = int(center_y - ear_height / 2)
    y2 = int(center_y + ear_height / 2)
    ear_roi = frame[y1:y2, x1:x2]

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(ear_roi, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Plot the edges on the original frame
    frame[y1:y2, x1:x2][edges != 0] = [0, 0, 255]


def plot_orientation(frame, center_x, center_y, ear_width, ear_height):
    # Calculate the coordinates of the top-left and bottom-right corners of the bounding box
    x1 = int(center_x - ear_width / 2)
    y1 = int(center_y - ear_height / 2)
    x2 = int(center_x + ear_width / 2)
    y2 = int(center_y + ear_height / 2)

    # Extract the ear region from the frame
    ear = frame[y1:y2, x1:x2]

    # Convert the ear region to grayscale
    ear_gray = cv2.cvtColor(ear, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to the grayscale image
    edges = cv2.Canny(ear_gray, 50, 150)

    # Find the contours in the edge image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour and fit a line to it
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)

        # Fit a line to the largest contour
        [vx, vy, x, y] = cv2.fitLine(max_contour, cv2.DIST_L2, 0, 0.01, 0.01)

        # Calculate the angle between the line and a vertical line passing through the center of the ear
        angle = np.arctan2(vx, vy) * 180 / np.pi

        # Draw a line representing the orientation of the ear
        pt1 = (int(x - vx * 50), int(y - vy * 50))
        pt2 = (int(x + vx * 50), int(y + vy * 50))
        cv2.line(frame, (x1 + pt1[0], y1 + pt1[1]), (x1 + pt2[0], y1 + pt2[1]), (0, 255, 0), 2)

        # Draw the contour of the ear region for reference
        cv2.drawContours(frame, [max_contour + np.array([[x1, y1]])], 0, (0, 0, 255), 2)


def plot_angle(x,y,w,h):
    # Calculate the center of the ear bounding box
    center = (int(x + w / 2),int(y + h / 2))

    # Calculate the angle of the ear bounding box
    angle = np.arctan2(y - center[1], x - center[0])

    # Plot a line from the center of the ear bounding box to a point on the edge of the ear bounding box at the specified angle
    cv2.line(frame, center, (int(center[0] + w * np.cos(angle)), int(center[1] + w * np.sin(angle))), color=(0, 0, 255),
             thickness=2)

if __name__ == '__main__':
    detector = Detector()
    cap = cv2.VideoCapture(0)

    # Initialize variables for tracking ear orientation and center
    prev_angle = None
    center_x = None
    center_y = None
    ear_width = None
    ear_height = None

    while True:
        ret, frame = cap.read()
        detected_loc = detector.detect(frame)
        for x, y, w, h in detected_loc[:1]:
            center_x = x + w / 2
            center_y = y + h / 2
            ear_width = w
            ear_height = h

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 4)
            cv2.line(frame, (int(center_x), 0), (int(center_x), frame.shape[0]), (0, 255, 0), 2)
            cv2.line(frame, (0, int(center_y)), (frame.shape[1], int(center_y)), (0, 255, 0), 2)

            # Plot the helix of the ear
            # if center_x is not None and center_y is not None and ear_width is not None and ear_height is not None:
            #     plot_helix(frame, center_x, center_y, ear_width, ear_height)

            plot_edges(frame, center_x, center_y, ear_width, ear_height)
            # plot_orientation(frame, center_x, center_y, ear_width, ear_height)
            plot_angle(x,y,w,h)

        cv2.imshow('Ear Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close all windows
    cap.release()
    cv2.destroyAllWindows()
