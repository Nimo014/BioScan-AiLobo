import cv2

# Load the HOG + Linear SVM eye detection model
model = cv2.HOGDescriptor()
model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open the video stream
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for input to the model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes using the HOG + Linear SVM model
    eyes, _ = model.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # Loop over the detected eyes
    for (x, y, w, h) in eyes:
        # Compute the center point of the eye
        center = (int(x + 0.5*w), int(y + 0.5*h))

        # Draw a dot at the center of the eye
        cv2.circle(frame, center, 2, (0, 0, 255), -1)

    # Show the output frame
    cv2.imshow('Frame', frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
