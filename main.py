#!/usr/bin/python
# -*- coding:utf-8 -*-
import cv2

import contour.find_contour as fc
from contour.calculate_counter import CalculationContour
from face_detection.face_mask import FaceMask
from face_detection.face_rec import FaceIdentify
from port.serial_port import SerialPort

import numpy as np

HAND_CAMERA_ID = 1  # 0为（笔记本）内置摄像头，台式机输入0
FACE_CAMERA_ID = 3
PORT_ID = "COM15"  # 串口端口号

LOWER = np.array([70, 138, 86], np.uint8)
UPPER = np.array([168, 160, 123], np.uint8)

FACE_CAP = cv2.VideoCapture(FACE_CAMERA_ID)
HAND_CAP = cv2.VideoCapture(HAND_CAMERA_ID)


class Main:

    def __init__(self):
        self._init_show_pic()
        self._init_tools()

    def _init_show_pic(self):
        """
        初始化需要显示的图片
        """
        self.original_frame = None
        self.black_white_frame = None
        self.draw_frame = None

    def _init_tools(self):
        """
        引入计算和分析工具的实例
        """
        self.serial_port = SerialPort(PORT_ID)
        self.face_mask = FaceMask()
        self.face_rec = FaceIdentify()
        self.calculate_counter = CalculationContour()

    def _init_last_value(self):
        """
        初始化数据点的防抖，需要记录历史数据
        """
        self.last_wrist_point = (-1, -1)
        self.last_palm_calculate = (-1, -1)
        self.last_if_hand = False

    def _send_data(self):
        if abs(self.last_wrist_point[0] - self.hand_point_data["wrist_point"][0]) >= 26 or \
                abs(self.last_wrist_point[1] - self.hand_point_data["wrist_point"][1]) >= 26:
            self.last_test_hand = self.hand_point_data["wrist_point"]
            self.serial_port.send("$hx%07dy%07d^" % (self.last_test_hand[0], self.last_test_hand[1]))
        if abs(self.last_palm_calculate[0] - self.hand_point_data["palm_point"][0]) >= 26 or \
                abs(self.last_palm_calculate[1] - self.hand_point_data["palm_point"][1]) >= 26:
            self.last_palm_calculate = self.hand_point_data["palm_point"]

            self.serial_port.send("$wx%07dy%07d^" % (self.last_palm_calculate[0], self.last_palm_calculate[1]))

    def _cv2_show(self):
        """
        显示图片
        """
        cv2.imshow("original_frame", self.original_frame)
        cv2.imshow("black_white_frame", self.black_white_frame)
        cv2.imshow("draw_frame", self.draw_frame)

    def _main_hand_rec(self, frame):

        self.original_frame = frame
        self.black_white_frame, self.color_contour = fc.pre_cut_color(
            frame, LOWER, UPPER)
        self.ostu_counter = fc.Interception_contour(fc.skin_mask(frame))
        try:
            self.hand_point_data = self.calculate_counter.main_calculate(
                self.ostu_counter)
            self._draw_hand_frame()
            self._send_data()
        except ValueError:
            try:
                self.ostu_counter = fc.Interception_contour(fc.skin_mask(frame[0:420, 0:640]))
                self.hand_point_data = self.calculate_counter.main_calculate(self.ostu_counter)
                self._draw_hand_frame()
                self._send_data()
            except ValueError:
                self.draw_frame = self.original_frame
        self._cv2_show()

    def _draw_hand_frame(self):
        self.draw_frame = self.original_frame.copy()
        cv2.drawContours(self.original_frame, self.ostu_counter, -1, (0, 255, 0), 3)
        cv2.circle(self.draw_frame, self.hand_point_data["palm_point"], self.hand_point_data["max_distance"],
                   (255, 0, 242), 4)
        for point in self.hand_point_data["draw_point_list"]:
            cv2.circle(self.draw_frame, point, 1, (0, 0, 255), 4)

    def main_face_rec(self, frame):
        pass


