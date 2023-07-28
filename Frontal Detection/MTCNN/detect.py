import cv2
import numpy as np
from mtcnn import MTCNN
import time
# create MTCNN detector
detector = MTCNN()

# create video capture object
cap = cv2.VideoCapture(0)


fps_start_time = time.time()
fps = 0
frame_counter = 0
while True:
    # read frame from camera
    ret, frame = cap.read()

    if not ret:
        break

    # detect faces and landmarks
    results = detector.detect_faces(frame)

    # iterate through each face found
    for result in results:
        # extract the bounding box and landmarks
        bbox = result['box']
        landmarks = result['keypoints']

        # extract coordinates of left and right eyes and nose
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        nose = landmarks['nose']
        mouth_left = landmarks['mouth_left']
        mouth_right = landmarks['mouth_right']
        mouth_mid = ((mouth_left[0] + mouth_right[0])//2, (mouth_left[1] + mouth_right[1])//2)

        # draw bounding box and landmarks on frame
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        cv2.circle(frame, left_eye, 2, (0, 0, 255), 2)
        cv2.circle(frame, right_eye, 2, (0, 0, 255), 2)
        cv2.circle(frame, nose, 2, (0, 0, 255), 2)
        #cv2.circle(frame, mouth_mid, 2, (0, 0, 255), 2)
    #cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # display the frame with detected facial landmarks
    cv2.imshow("Facial Landmarks", frame)


    # break the loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
