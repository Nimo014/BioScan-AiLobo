import cv2
import numpy as np

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, image = cap.read()
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform circle detection using Hough Transform
    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=50, param2=30, minRadius=10, maxRadius=50
    )

    # Check if circles were detected
    if circles is not None:
        # Convert the (x, y, r) coordinates and radius to integers
        circles = np.round(circles[0, :]).astype("int")

        # Iterate over the detected circles
        for (x, y, r) in circles:
            # Draw the circle on the image
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)

        cv2.imshow("Ear Detection", image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.waitKey(0)
cv2.destroyAllWindows()
