import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection and Face Mesh models
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open a video capture object to acquire video frames from the camera

url = 'http://192.168.1.7:8080/video'
cap = cv2.VideoCapture(0)

while True:
    # Capture a video frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    result_detection = face_detection.process(frame_rgb)
    if result_detection.detections:
        for detection in result_detection.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Convert the frame to RGB and pass it to MediaPipe Face Mesh model
            result_mesh = face_mesh.process(frame_rgb)
            if result_mesh.multi_face_landmarks:
                for face_landmarks in result_mesh.multi_face_landmarks:
                    roi_landmark_1 = face_landmarks.landmark[447]
                    roi_landmark_2 = face_landmarks.landmark[366]
                    roi_landmark_1 = [roi_landmark_1, roi_landmark_2]
                    for landmark in roi_landmark_1:
                        print(landmark)
                        x_pixel, y_pixel = int((landmark.x) * iw), int((landmark.y) * ih)

                        x_pixel_2, y_pixel_2 = int((landmark.x+0.05)* iw), int((landmark.y-0.05) * ih)
                        cv2.circle(frame, (x_pixel, y_pixel), 3, (0, 255, 0), -1)
                        cv2.circle(frame, (x_pixel_2, y_pixel_2), 3, (0, 255, 0), -1)

    # Show the frame with the plotted landmarks
    cv2.imshow("Facial Landmarks", frame)

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
