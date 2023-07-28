import cv2

# Load the pre-trained Haar cascade for upper body detection
upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Function to detect and draw bounding boxes around upper bodies
def detect_upper_bodies(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect upper bodies in the image
    upper_bodies = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 100))

    # Draw bounding boxes around the detected upper bodies
    for (x, y, w, h) in upper_bodies:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image

# Open a video capture object
video_capture = cv2.VideoCapture(0)

while True:
    # Capture video frame-by-frame
    ret, frame = video_capture.read()

    # Detect upper bodies in the frame
    result_frame = detect_upper_bodies(frame)

    # Display the resulting frame
    cv2.imshow('Upper Bodies Detection', result_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
