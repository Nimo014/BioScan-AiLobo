import os
import cv2
import torch
from convert_annotations import denormalize
import math
import numpy as np


class Detector:
    weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), '', 'best.pt')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

    # By calling with image_name, the model resizes the picture as appropriate
    def detect(self, image_name):
        detections = []
        results = self.model(image_name, size=416)
        for tensor1, tensor2 in zip(results.xywhn, results.xywh):
            for result1, result2 in zip(tensor1, tensor2):
                x_norm, y_norm, w_norm, h_norm, _, _ = result1.numpy()
                x_, y_, w_, h_, _, _ = result2.numpy()
                detections.extend([denormalize(x_norm, y_norm, w_norm, h_norm)])

                # denormalize(results.xywh) != results.xywh !!
                # print(denormalize(x_norm, y_norm, w_norm, h_norm))
                # print(x_, y_, w_, h_)

        return detections


if __name__ == '__main__':
    detector = Detector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        #cv2.imwrite('captured_image.jpg',image)

        detected_loc = detector.detect(frame)
        for x, y, w, h in detected_loc[:1]:
            center_x = x + w / 2
            center_y = y + h / 2

            # Get the angle of rotation based on the aspect ratio of the bounding box
            aspect_ratio = w / float(h)
            angle = math.atan(aspect_ratio) * 180 / math.pi

            # Rotate the bounding box around the center point by the calculated angle
            box = cv2.boxPoints(((center_x, center_y), (w, h), angle))
            box = np.int0(box)

            # Draw a line from the center point to the top of the rotated bounding box
            line_length = h / 2
            line_end_x = center_x - line_length * math.sin(math.radians(angle))
            line_end_y = center_y + line_length * math.cos(math.radians(angle))
            cv2.line(frame, (int(center_x), int(center_y)), (int(line_end_x), int(line_end_y)), (255, 0, 0), 2)

            # Draw a rotated rectangle around the bounding box
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

            # Display the output frame
        cv2.imshow('Ear Detector', frame)

            #cv2.rectangle(image, (x, y), (x + w, y + h), (128, 255, 0), 4)
        #cv2.imshow('Ear Detector', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
