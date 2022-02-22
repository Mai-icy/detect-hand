#!/usr/bin/python
# -*- coding:utf-8 -*-
from contour.find_contour import pre_cut_color
import cv2
import sys

import numpy as np

HAND_CAMERA_ID = 1  # 0为（笔记本）内置摄像头，台式机输入0
FACE_CAMERA_ID = 0

FACE_CAP = cv2.VideoCapture(FACE_CAMERA_ID)
HAND_CAP = cv2.VideoCapture(HAND_CAMERA_ID)


LOWER = np.array([70, 138, 86], np.uint8)
UPPER = np.array([168, 160, 123], np.uint8)


while True:
    _, face_frame = FACE_CAP.read()
    _, hand_frame = HAND_CAP.read()
    cv2.imshow("face", face_frame)
    cv2.imshow("hand", hand_frame)

    b_w, b_c = pre_cut_color(hand_frame, LOWER, UPPER)

    cv2.imshow("b_w", b_w)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        cv2.destroyAllWindows()
        sys.exit(0)


