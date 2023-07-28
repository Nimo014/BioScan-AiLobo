import cv2
import numpy as np

def detect_back_of_head(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection (Canny or Sobel)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)  # Adjust the thresholds based on your image

    # Select the region of interest (ROI) where the head is expected
    roi = edges[100:400, 100:500]  # Adjust the ROI coordinates based on your image

    # Find contours in the ROI
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]  # Adjust the area threshold as needed

    # Estimate the back of the head contour based on its shape and position
    back_of_head_contour = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        if aspect_ratio < 0.8:  # Adjust the aspect ratio threshold as needed
            back_of_head_contour = cnt
            break

    if back_of_head_contour is not None:
        # Calculate the center of the back of the head
        M = cv2.moments(back_of_head_contour)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])

        # Adjust the coordinates based on the ROI position
        center_x += 100
        center_y += 100

        return (center_x, center_y)

    return None

# Initialize the video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Detect the center of the back of the head
    center = detect_back_of_head(frame)

    if center is not None:
        # Draw a dot at the center of the back of the head
        cv2.circle(frame, center, 5, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow('Back of Head Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
