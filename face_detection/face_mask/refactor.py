#!/usr/bin/python
# -*- coding:utf-8 -*-
# 源码地址： https://github.com/bubbliiiing/mask-recognize
# 经过了结构优化和整理
import cv2
import numpy as np
from keras.applications.imagenet_utils import preprocess_input

import utils.utils as utils
from net.mobilenet import MobileNet
from net.mtcnn import Mtcnn


class FaceMask:
    def __init__(self):
        self.threshold = [0.5, 0.6, 0.8]  # 阈值参数
        self._init_ai()

    def _init_ai(self):
        """
        初始化数据集模型，以及人脸识别模型
        """
        self.mtcnn_model = Mtcnn()  # 人脸识别mtcnn模型
        self.Crop_HEIGHT = 224
        self.Crop_WIDTH = 224
        self.NUM_CLASSES = 2
        self.mask_model = MobileNet(
            input_shape=[
                self.Crop_HEIGHT,
                self.Crop_WIDTH,
                3],
            classes=self.NUM_CLASSES)
        self.mask_model.load_weights(__file__ + "/logs/last_one.h5")

    def _load_frame(self, frame):
        """
        载入图片，并获取它的基本数据到类成员以供处理。必须要在其他处理函数(_judge_face， _judge_masks)之前
        :param frame: 原图片
        """
        self.height, self.width, _ = np.shape(frame)
        self.draw_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    def _judge_face(self, *, is_single=True):
        """
        判断图片中是否有人脸，图片会从类成员中获取，_load_frame会载入图片
        :param is_single: 是否只获取最大的单张人脸
        :return: 如果有，返回rectangles，即人脸位置的方框数据。否则，返回None
        """
        def rectangle_area(rectangle):
            left, top, right, bottom = rectangle[0:4]
            return (top - bottom) * (right - left)
        #   检测人脸
        rectangles = self.mtcnn_model.detectFace(self.draw_rgb, self.threshold)
        rectangles = np.array(rectangles, dtype=np.int32)
        if rectangles.any():
            if is_single:
                return np.array(sorted(rectangles, key=rectangle_area)[0])[None]
            else:
                return rectangles
        else:
            return None

    def _judge_masks(self, rectangles):
        """
        把获取到的人脸坐标集（rectangles）进行口罩识别，传入要进行判断的rectangles，返回一一对应的True或False列表
        （图片会从类成员中获取，_load_frame会载入图片）
        :param rectangles: 由_get_face成员函数返回的数据
        :return: True或False列表
        """
        rectangles_temp = utils.rect2square(rectangles)
        rectangles_temp[:, [0, 2]] = np.clip(
            rectangles_temp[:, [0, 2]], 0, self.width)
        rectangles_temp[:, [1, 3]] = np.clip(
            rectangles_temp[:, [1, 3]], 0, self.height)
        # 对检测到的人脸进行编码
        classes_all = []
        for rectangle in rectangles_temp:
            landmark = np.reshape(
                rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])
            crop_img = self.draw_rgb[int(rectangle[1]):int(
                rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            #   利用人脸关键点进行人脸对齐
            crop_img, _ = utils.Alignment_1(crop_img, landmark)
            crop_img = cv2.resize(
                crop_img, (self.Crop_WIDTH, self.Crop_HEIGHT))
            crop_img = preprocess_input(
                np.reshape(
                    np.array(
                        crop_img, np.float64), [
                        1, self.Crop_HEIGHT, self.Crop_WIDTH, 3]))

            classes = [True, False][np.argmax(
                self.mask_model.predict(crop_img)[0])]
            classes_all.append(classes)
            return classes_all

    @staticmethod
    def _draw_frame(frame, rectangles, judge_res):
        """
        传入结果，进行绘图，并将处理后的图片返回
        :param frame: 原图片
        :param rectangles: 由_get_face成员函数返回的数据
        :param judge_res: 由_judge_masks成员函数返回的数据
        :return: 处理后的图片
        """
        rectangles = rectangles[:, 0:4]
        # 画出方框以及提示字
        test_list = [str(x) for x in judge_res]
        for (left, top, right, bottom), c in zip(rectangles, test_list):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, c, (left, bottom - 15),
                        font, 0.75, (255, 255, 255), 2)
        return frame

    def main(self, frame):  # 提供可修改的接口
        """
        集合函数处理图片以及返回结果
        :param frame: 传入的图片
        :return: 返回绘画之后的 图片 以及 判断值列表
        """
        self._load_frame(frame)
        face_rectangles = self._judge_face(is_single=True)
        if face_rectangles is not None:
            # 有人脸的获取以及判断状态
            judge_res = self._judge_masks(face_rectangles)
            res_frame = self._draw_frame(frame, face_rectangles, judge_res)
            return res_frame, judge_res
        else:
            # 没有人脸的空置状态
            res_frame = frame
            return res_frame, [False]


if __name__ == "__main__":
    test = FaceMask()

    video_capture = cv2.VideoCapture(0)
    while True:
        ret, draw = video_capture.read()

        draw, flag_list = test.main(draw)
        print(flag_list)

        cv2.imshow('Video', draw)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
