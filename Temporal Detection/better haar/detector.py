import cv2

# Load the Haar cascade classifier for ear detection
ear_cascade = cv2.CascadeClassifier(r'C:\Users\nirav\Documents\BioScan\Temporal_Detection\better haar\cascade_lateral_ears_opencv.xml')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ears in the frame using the Haar cascade classifier
    ears = ear_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Draw rectangles around the detected ears
    for (x, y, w, h) in ears:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame with detected ears
    cv2.imshow('Ear Detection', frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
