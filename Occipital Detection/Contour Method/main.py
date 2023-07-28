import cv2
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture(0)  # 0 represents the default camera

while True:
    # Read the frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to improve the contour
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to find the back of the head
    filtered_contours = []
    for contour in contours:
        # Filter based on contour properties (you can adjust these criteria)
        area = cv2.contourArea(contour)
        if area > 100000:
            filtered_contours.append(contour)

    # Find the largest bounding box
    largest_box = None
    max_area = 0
    for contour in filtered_contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        area = cv2.contourArea(box)
        if area > max_area:
            max_area = area
            largest_box = box

    # Converge the box into the largest possible box
    if largest_box is not None:
        x, y, w, h = cv2.boundingRect(largest_box)
        largest_box = np.float32([[x, y], [x + w, y], [x + w, y + w], [x, y + w]])

        # Draw the largest box on the frame
        cv2.drawContours(frame, [largest_box.astype(int)], 0, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Real-time Head Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
