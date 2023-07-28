import cv2
import numpy as np

# Open the video capture
video_capture = cv2.VideoCapture(0)  # Change to the appropriate video source if needed

while True:
    # Read the current frame
    ret, frame = video_capture.read()

    # Check if frame was read successfully
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Sobel edge detection along the vertical axis
    edges = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.convertScaleAbs(edges)

    # Threshold the edges to obtain a binary image
    _, thresholded = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)

    # Create a green-colored image for displaying vertical edges
    green_edges = cv2.merge([np.zeros_like(thresholded), thresholded, np.zeros_like(thresholded)])

    # Display the green-colored edges in real-time
    cv2.imshow("Vertical Edges", green_edges)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
video_capture.release()
cv2.destroyAllWindows()
