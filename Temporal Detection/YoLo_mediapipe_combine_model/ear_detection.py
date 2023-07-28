import os

import cv2
import torch

from convert_annotations import denormalize
import pickle

class Detector:
    # weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), '', 'best.pt')
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
    #
    # # Save the model using pickle
    # with open('model.pkl', 'wb') as f:
    #     pickle.dump(model, f)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    # By calling with image_name, the model resizes the picture as appropriate
    def detect(self, image_name,):
        detections = []
        results = self.model(image_name, size=416)
        for tensor1, tensor2 in zip(results.xywhn, results.xywh):
            for result1, result2 in zip(tensor1, tensor2):
                x_norm, y_norm, w_norm, h_norm, _, _ = result1.numpy()
                x_, y_, w_, h_, _, _ = result2.numpy()
                x, y, w, h = denormalize(x_norm, y_norm, w_norm, h_norm)
                detections.append((x, y, w, h))

        return detections


if __name__ == '__main__':
    detector = Detector()
    url = 'http://192.168.1.52:4747/video'
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imwrite('captured_image.jpg', frame)

        detected_loc = detector.detect(frame)

        # # Reference line
        # cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (255, 255, 255), 2,
        #          cv2.LINE_AA)

        for x, y, w, h in detected_loc:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 4)
            # dots

            cv2.circle(frame, (x, int(y + h * 0.25)), 4, (0, 255, 0), 2)
            cv2.circle(frame, (int(x - w * 0.3), int(y + h * 0.25)), 4, (0, 255, 0), 2)
            cv2.putText(frame, 'Ear Detected ', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # cv2.putText(frame, 'Right Ear ', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, 'left Ear', (frame.shape[0], 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Left Ear Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
