import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imwrite(r'C:\\Users\\nirav\\Documents\\BioScan\\Frontal_Detection\\images\\test_img.jpg', gray)

# Create an image with a line
img = np.zeros((512, 512, 3), np.uint8)
cv2.line(img, (100, 100), (400, 400), (0, 0, 255), 3)

# Get the line endpoints
x1, y1 = 100, 100
x2, y2 = 400, 400

# Compute the midpoint of the line
mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2

# Compute the slope of the line
slope = (y2 - y1) / (x2 - x1)

# Compute the slope of the perpendicular line
perp_slope = -1 / slope

# Compute the endpoints of the perpendicular line
perp_x1 = mid_x - 100
perp_y1 = mid_y - int(perp_slope * 100)
perp_x2 = mid_x + 100
perp_y2 = mid_y + int(perp_slope * 100)

# Draw the perpendicular line
cv2.line(img, (perp_x1, perp_y1), (perp_x2, perp_y2), (255, 0, 0), 3)

# Display the image
cv2.imshow("Perpendicular Line", img)
cv2.waitKey(0)

"""
import math

# calculate the angle between the eyes
eye_angle = math.atan2(landmarks.part(45).y - landmarks.part(36).y, landmarks.part(45).x - landmarks.part(36).x)

# calculate the midpoint of the nose
nose_midpoint = ((landmarks.part(31).x + landmarks.part(35).x) // 2, (landmarks.part(31).y + landmarks.part(35).y) // 2)

# calculate the position of the dots above the eyes
left_eye_dot = (landmarks.part(37).x - int(0.02 * math.cos(eye_angle)), landmarks.part(37).y - int(0.02 * math.sin(eye_angle)))
right_eye_dot = (landmarks.part(44).x - int(0.02 * math.cos(eye_angle)), landmarks.part(44).y - int(0.02 * math.sin(eye_angle)))

# calculate the position of the dot equidistant from the midpoint of the nose
nose_dot = (nose_midpoint[0], nose_midpoint[1] - int(0.25 * math.sqrt((landmarks.part(42).x - landmarks.part(39).x) ** 2 + (landmarks.part(42).y - landmarks.part(39).y) ** 2)))

# draw the dots on the image
cv2.circle(image, left_eye_dot, 3, (0, 255, 0), -1)
cv2.circle(image, right_eye_dot, 3, (0, 255, 0), -1)
cv2.circle(image, nose_dot, 3, (0, 255, 0), -1)

"""
