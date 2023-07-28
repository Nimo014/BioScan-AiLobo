import os
import cv2
import torch
from convert_annotations import denormalize


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
        cv2.imwrite('captured_image.jpg',frame)

        detected_loc = detector.detect(frame)
        for x, y, w, h in detected_loc:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 255, 0), 4)
        cv2.imshow('Ear Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
