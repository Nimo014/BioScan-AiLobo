import cv2
import numpy as np

# Step 1: Edge Detection
def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

# Step 2: Scan Rows and Calculate Lengths
def calculate_lengths(edges):
    lengths = []
    for row in edges:
        white_pixels = np.where(row == 255)[0]
        if len(white_pixels) > 0:
            length = white_pixels[-1] - white_pixels[0]
            lengths.append(length)
    return lengths

# Step 3: Find Point C
def find_point_c(lengths):
    diff = np.diff(lengths)
    if len(diff) > 0:
        c_index = np.argmax(diff)
        return c_index
    else:
        return None

# Step 4: Shoulder Point Detection
def detect_shoulder_points(lengths, c_index):
    left_lengths = lengths[:c_index]
    right_lengths = lengths[c_index:]

    left_slope_diff = np.diff(left_lengths)
    right_slope_diff = np.diff(right_lengths)

    left_shoulder_index = np.argmax(left_slope_diff)
    right_shoulder_index = np.argmax(right_slope_diff)

    return left_shoulder_index, right_shoulder_index

# Video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the camera
    ret, frame = cap.read()

    # Step 1: Edge Detection
    edges = detect_edges(frame)

    # Step 2: Scan Rows and Calculate Lengths
    lengths = calculate_lengths(edges)

    # Step 3: Find Point C
    c_index = find_point_c(lengths)

    # Step 4: Shoulder Point Detection
    left_shoulder_index, right_shoulder_index = detect_shoulder_points(lengths, c_index)

    # Display the results
    cv2.imshow('Edges', edges)
    cv2.imshow('Frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
