import cv2
import dlib
import time

# Replace the IP address and port number with your phone's IP address and port number
url = 'http://192.168.1.148:8080/video'
cap = cv2.VideoCapture(url)

# Set the desired width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('dlib/shape_predictor_68_face_landmarks.dat')

fps_start_time = time.time()
fps = 0
frame_counter = 0

while True:
    ret, img = cap.read()
    if not ret:
        print('Error fetching frame')
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces using the dlib detector
    faces = detector(gray, 0)
    #
    for face in faces:
        # Get the facial landmarks for the face
        landmarks = predictor(gray, face)

        # Draw a circle at each landmark point
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

        # Draw a rectangle around the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    if time_diff > 0:
        fps = int(frame_counter/time_diff)
        frame_counter = 0

    cv2.putText(img, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Phone Camera', img)

    if cv2.waitKey(1) == ord('q'):
        break

    frame_counter += 1
    fps_start_time = time.time()

cap.release()
cv2.destroyAllWindows()
