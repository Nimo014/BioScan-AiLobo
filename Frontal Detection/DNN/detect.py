import cv2
import numpy as np

# Load the pre-trained model
model_path = 'models/deploy.prototxt'
weights_path = 'models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
net = cv2.dnn.readNetFromCaffe(model_path, weights_path)

# Open the video stream
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for input to the model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the preprocessed frame through the model
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.1:
            # Compute bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            startX, startY, endX, endY = box.astype('int')


            # Draw the face bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

            # Extract the coordinates of the eyes
            left_eye_start = (int(detections[0, 0, i, 8] * frame.shape[1]), int(detections[0, 0, i, 9] * frame.shape[0]))
            left_eye_end = (int(detections[0, 0, i, 10] * frame.shape[1]), int(detections[0, 0, i, 11] * frame.shape[0]))
            right_eye_start = (int(detections[0, 0, i, 12] * frame.shape[1]), int(detections[0, 0, i, 13] * frame.shape[0]))
            right_eye_end = (int(detections[0, 0, i, 14] * frame.shape[1]), int(detections[0, 0, i, 15] * frame.shape[0]))

            # Calculate the center of the eyes
            left_eye_center = ((left_eye_start[0] + left_eye_end[0]) // 2, (left_eye_start[1] + left_eye_end[1]) // 2)
            right_eye_center = ((right_eye_start[0] + right_eye_end[0]) // 2, (right_eye_start[1] + right_eye_end[1]) // 2)

            # Draw the dots in the center of the eyes
            cv2.circle(frame, left_eye_center, 2, (0, 0, 255), -1)
            cv2.circle(frame, right_eye_center, 2, (0, 0, 255), -1)

    # Show the output frame
    cv2.imshow('Frame', frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
