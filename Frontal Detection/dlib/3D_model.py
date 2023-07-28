import cv2
import dlib
import numpy as np

# Load the pre-trained facial landmark detection model
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the camera and start capturing video frames
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    if not ret:
        break

    # Detect faces in the input image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray, 0)

    # Loop over the detected faces
    for face in faces:
        # Get the facial landmarks
        landmarks = predictor(gray, face)

        # Estimate the head pose
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),       # Chin
            (-225.0, 170.0, -135.0),    # Left eye left corner
            (225.0, 170.0, -135.0),     # Right eye right corner
            (-150.0, -150.0, -125.0),   # Left Mouth corner
            (150.0, -150.0, -125.0)     # Right mouth corner
        ])
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),     # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),       # Chin
            (landmarks.part(36).x, landmarks.part(36).y),     # Left eye left corner
            (landmarks.part(45).x, landmarks.part(45).y),     # Right eye right corner
            (landmarks.part(48).x, landmarks.part(48).y),     # Left Mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)      # Right mouth corner
        ])

        # Camera parameters
        focal_length = img.shape[1]
        center = (img.shape[1]/2, img.shape[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype='double'
        )

        # Assume no distortion
        dist_coeffs = np.zeros((4,1))

        # Solve for the rotation and translation vectors of the camera
        _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

        # Project the 3D model points onto the 2D image plane
        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        # Draw a line connecting the nose tip and the dot above the eye
        dot_pos = (landmarks.part(29).x, landmarks.part(29).y - 20)  # 20 pixels above the nose tip
        cv2.line(img, (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])), dot_pos, (0, 255, 0), 2)

    # Show the output image
    cv2.imshow('Output', img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()