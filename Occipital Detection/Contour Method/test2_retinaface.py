import cv2
from retinaface import RetinaFace

# Initialize the video capture
cap = cv2.VideoCapture(0)  # 0 represents the default camera

# Initialize RetinaFace for face detection
detector = RetinaFace(quality='normal')

while True:
    # Read the frame from the video capture
    ret, frame = cap.read()

    # Detect faces using RetinaFace
    faces = detector.detect(frame)

    # Iterate over detected faces
    for face in faces:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = int(face['box'][0]), int(face['box'][1]), int(face['box'][2]), int(face['box'][3])

        # Draw the bounding box around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the label 'Back of Head' near the bounding box
        cv2.putText(frame, 'Back of Head', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Real-time Head Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
