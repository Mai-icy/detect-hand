#!/usr/bin/python
# -*- coding:utf-8 -*-
import cv2


class GetFace:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance:
            return cls._instance
        else:
            return super.__new__(*args, **kwargs)

    def __init__(self):
        """
        载入文件 OpenCV的级联分类器
        """
        file_path = "haarcascade_frontalface_default.xml"
        self.face_CascadeClassifier = cv2.CascadeClassifier(file_path)
        self.face_CascadeClassifier.load(file_path)
        self.time_delay = 0

    def judge_face(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = self.face_CascadeClassifier.detectMultiScale(gray, 1.1, 3, 0, (200, 200))
        for (x, y, w, h) in faces:
            face_img = cv2.rectangle(
                img, (x - 20, y - 20), (x + w + 20, y + h + 20), (255, 255, 0), 2)
            cv2.imshow('1', face_img)
            face_frame_output = img[y - 20:y + h + 20, x - 20:x + h + 20]
            if len(faces) == 1:  # 长度>1会出现同框的多个人脸，所以限定为1的时候截取，目前代码为空
                return True
        return False



