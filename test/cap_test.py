#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys

import cv2


HAND_CAMERA_ID = 1  # 0为（笔记本）内置摄像头，台式机输入0
FACE_CAMERA_ID = 0

FACE_CAP = cv2.VideoCapture(FACE_CAMERA_ID)
HAND_CAP = cv2.VideoCapture(HAND_CAMERA_ID)


while True:
    _, face_frame = FACE_CAP.read()
    _, hand_frame = HAND_CAP.read()
    cv2.imshow("face", face_frame)
    cv2.imshow("hand", hand_frame)
    cv2.imshow("test", hand_frame[0:420, 0:640])

    print(hand_frame.shape[0], hand_frame.shape[1])

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        cv2.destroyAllWindows()
        sys.exit(0)



