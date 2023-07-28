import cv2
import numpy as np
import os
import sys


class Detector:

	cascade_left = cv2.CascadeClassifier(
		os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_leftear.xml'))

	cascade_right = cv2.CascadeClassifier(
		os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_rightear.xml'))

	def detect(self, img_):
		det_list_left = self.cascade_left.detectMultiScale(img_, 1.5, 3)
		det_list_right = self.cascade_right.detectMultiScale(img_, 1.05, 5)

		if type(det_list_right) is not tuple and type(det_list_left) is not tuple:
			return np.concatenate((det_list_left, det_list_right), axis=0)
		if type(det_list_right) is not tuple:
			return det_list_right
		else:
			return det_list_left


if __name__ == '__main__':
	detector = Detector()
	cap = cv2.VideoCapture(0)

	while True:
		ret, frame = cap.read()
		detected_loc = detector.detect(frame)

		for x, y, w, h in detected_loc:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 255, 0), 4)

		cv2.imshow('Face Detector', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()