# from cv2 import CAP_V4L2
import time
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf

# ---setting------------------------------#

# recording setting
recording = False
recording_time = 25

# model input size
input_size = 200  # 200

# ear model
landmark_size = 55
ear_part_num = [20, 15, 15, 5]
ear_threshold = 0.5

# resize setting for coordinate correction
r_size = 200  # 200

# capture size (Default = 620x480)
IM_W = 1280
IM_H = 720

# output size (Default = 620x480)
o_size_w = 720
o_size_h = 720


# ----------------------------------------#

def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def apply_blur(img, landmark):
    blur = _gaussian_kernel(5, 2.5, landmark, img.dtype)
    img = tf.nn.depthwise_conv2d(img, blur, [1, 1, 1, 1], 'SAME')
    return img[0]


color_list = [(0, 255, 0), (255, 51, 0), (255, 204, 0), (255, 204, 0)]

model = tf.keras.models.load_model('./saved_model_h5/saved_model_openpose_ear_v1.h5', compile=False)

pred = tf.keras.backend.function([model.input], [model.get_layer('s6').output])

if recording:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(datetime.today().strftime("%Y%m%d%H%M%S") + '.avi', fourcc, 25.0, (o_size_w, o_size_h))

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, IM_W)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, IM_H)

start = time.time()

cut_w_r = (IM_W - IM_H) // 2
cut_w_l = IM_W - cut_w_r

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()

    # frame = cv2.resize(frame, (o_size_w, o_size_h), interpolation = cv2.INTER_AREA)
    frame = frame[:, cut_w_r:cut_w_l, :]
    frame = cv2.flip(frame, 1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(input_size, input_size), interpolation=cv2.INTER_AREA)

    result_all = pred([np.expand_dims(image, axis=0) / 255.])
    # ear-detect---------------------------------------------------------------
    result = result_all[0]
    result[result < ear_threshold] = 0  # threshold setting
    result = tf.image.resize(result, [r_size, r_size])
    result = apply_blur(result, landmark_size).numpy()
    result = np.argmax(result.reshape(-1, landmark_size), axis=0)

    prev_xy = [[], [], [], []]
    for i, idx in enumerate(result):
        x, y = idx % r_size / r_size * o_size_w, idx // r_size / r_size * o_size_h
        if x < 1 or y < 1: continue
        if i > 49: prev_xy[3].append([int(x), int(y)]); continue
        if i > 34: prev_xy[2].append([int(x), int(y)]); continue
        if i > 19: prev_xy[1].append([int(x), int(y)]); continue
        # cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
        prev_xy[0].append([int(x), int(y)])

    for i, xy in enumerate(prev_xy):
        if len(xy) == ear_part_num[i]:
            cv2.polylines(frame, [np.asarray(xy)], False, color_list[i], 2)
    # -------------------------------------------------------------------------

    if ret and recording:
        if (time.time() - start) > recording_time: break
        out.write(frame)

    # guide line
    # cv2.ellipse(frame, (o_size_w//2,o_size_h//2), (o_size_w//5,o_size_h//3), 0, 0, 360, (0, 255, 0), 1)
    cv2.imshow("VFrame", frame)

capture.release()
if recording:
    out.release()
cv2.destroyAllWindows()
