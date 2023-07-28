import cv2
from mtcnn import MTCNN

url = 'http://192.168.1.8:4747/video'
# Initialize the video capture
cap = cv2.VideoCapture(0)  # 0 represents the default camera

# Initialize MTCNN for face detection
detector = MTCNN()

while True:
    # Read the frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using MTCNN
    faces = detector.detect_faces(frame)

    # Iterate over detected faces
    for face in faces:
        # Extract bounding box coordinates
        x, y, width, height = face['box']

        # Draw the bounding box around the face
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Display the label 'Back of Head' near the bounding box
        cv2.putText(frame, 'Back of Head', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Real-time Head Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
