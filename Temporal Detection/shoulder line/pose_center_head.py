import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize MediaPipe Pose module and video capture
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
url = 'http://192.168.1.148:4747/video'
cap = cv2.VideoCapture(url)  # Use 0 for webcam or provide the path to a video file

while True:
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a mirrored view
    image = cv2.flip(image, 1)

    # Convert the image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to get pose landmarks
    results = pose.process(image_rgb)

    # Check if pose landmarks are detected
    if results.pose_landmarks:
        # Retrieve shoulder landmarks
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        #neck = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NECK]

        # Convert landmark coordinates to pixel values
        h, w, _ = image.shape
        left_shoulder_x = int(left_shoulder.x * w)
        left_shoulder_y = int(left_shoulder.y * h)
        right_shoulder_x = int(right_shoulder.x * w)
        right_shoulder_y = int(right_shoulder.y * h)

        # Calculate shoulder center
        shoulder_center_x = int((left_shoulder_x + right_shoulder_x) / 2)
        shoulder_center_y = int((left_shoulder_y + right_shoulder_y) / 2)

        # Draw dots above the shoulder center
        dot_y = int(shoulder_center_y - 0.4 * abs(right_shoulder_x - left_shoulder_x))
        cv2.circle(image, (shoulder_center_x, dot_y), 1, (255, 0, 0), -1)

        # Draw line to visualize shoulder
        cv2.line(image, (left_shoulder_x, left_shoulder_y), (right_shoulder_x, right_shoulder_y), (0, 255, 0), 3)

        # Draw line to visualize shoulder
        cv2.line(image, (left_shoulder_x, left_shoulder_y), (right_shoulder_x, right_shoulder_y), (0, 255, 0), 3)

    # Display the image
    cv2.imshow('Shoulder Landmarks', image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
