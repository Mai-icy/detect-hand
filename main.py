#!/usr/bin/python
# -*- coding:utf-8 -*-
import cv2
import sys

import contour.find_contour as fc
from contour.calculate_counter import CalculationContour, WristError, CalculateError
from face_detection.face_mask import FaceMask
from face_detection.face_rec import FaceIdentify
from port.serial_port import SerialPort

import numpy as np
from PIL import Image

HAND_CAMERA_ID = 1  # 0为（笔记本）内置摄像头，台式机输入0
FACE_CAMERA_ID = 0
PORT_ID = "COM5"  # 串口端口号

LOWER = np.array([70, 138, 86], np.uint8)
UPPER = np.array([168, 160, 123], np.uint8)

FACE_CAP = cv2.VideoCapture(FACE_CAMERA_ID)
HAND_CAP = cv2.VideoCapture(HAND_CAMERA_ID)


class Main:

    def __init__(self):
        self._init_show_pic()
        self._init_tools()
        self._init_last_value()

    def _init_show_pic(self):
        """
        初始化需要显示的图片
        """
        self.original_frame = None
        self.black_white_frame = None
        self.draw_frame = None
        self.face_frame = None

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
        self.hand_point_data = {
            "palm_point": (-1, -1),
            "max_distance": (-1, -1),
            "wrist_point": (-1, -1),
            "draw_point_list": []
        }

    def _send_data(self):
        """
        发送数据到串口，在点数据具有较大变动时才发送。
        :return:
        """
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
        cv2.imshow("face_frame", self.face_frame)

    def _main_hand_rec(self, frame):
        """
        处理手识别的对外接口，会获取原图片的各个点数据，保存到成员变量。
        :param frame: 手摄像头获取的原图片。
        :return:
        """
        self.original_frame = frame.copy()
        self.black_white_frame, self.color_contour = fc.pre_cut_color(frame, LOWER, UPPER)
        try:
            color_area = cv2.contourArea(self.color_contour)
        except cv2.error:
            if self.last_if_hand:
                self.last_if_hand = False
                print("没有手")  # todo
            self.draw_frame = self.original_frame
            return
        if color_area < 11000:
            if self.last_if_hand:
                self.last_if_hand = False
                print("没有手")  # todo
            self.draw_frame = self.original_frame
            return
        if not self.last_if_hand:
            self.last_if_hand = True
            print("有手")  # todo

        cut_y = frame.shape[0]
        while True:
            try:
                cut_frame = frame[0:cut_y, 0:640]
                self.ostu_counter = fc.Interception_contour(fc.skin_mask(cut_frame))
                self.hand_point_data = self.calculate_counter.main_calculate(
                    self.ostu_counter, cut_y)
                self._draw_hand_frame()
                self.is_block = False
                break
            except CalculateError:
                cut_y -= 7
            except WristError:
                print("袖子遮挡")  # todo
                self.draw_frame = self.original_frame.copy()
                im_tip_pic = Image.open('pic\\pls_show_hand.png')
                image = Image.fromarray(cv2.cvtColor(self.draw_frame, cv2.COLOR_BGR2RGB))
                image.paste(im_tip_pic, (0, 0), im_tip_pic)
                self.draw_frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                break

    def _draw_hand_frame(self):
        """
        根据关键点数据进行绘画。
        :return:
        """
        self.draw_frame = self.original_frame.copy()
        cv2.drawContours(self.draw_frame, self.ostu_counter, -1, (0, 255, 0), 3)
        palm_point = self.hand_point_data["palm_point"]
        wrist_point = self.hand_point_data["wrist_point"]
        cv2.circle(self.draw_frame, palm_point, self.hand_point_data["max_distance"],
                   (255, 0, 242), 4)
        cv2.circle(self.draw_frame, wrist_point, 4, (255, 100, 100), 4)
        for point in self.hand_point_data["draw_point_list"]:
            point = (int(point[0]), int(point[1]))
            cv2.circle(self.draw_frame, point, 2, (0, 0, 255), 4)

        im_hand_pic = Image.open('pic\\hand_point1.png')
        im_wrist_pic = Image.open('pic\\wrist_point.png')

        image = Image.fromarray(cv2.cvtColor(self.draw_frame, cv2.COLOR_BGR2RGB))
        image.paste(im_hand_pic, (palm_point[0], palm_point[1] - im_hand_pic.height), im_hand_pic)
        image.paste(im_wrist_pic, (wrist_point[0], wrist_point[1] - im_wrist_pic.height), im_wrist_pic)

        self.draw_frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def _main_face_rec(self, frame):
        self.face_frame = frame

        self.face_frame, flag_list = self.face_mask.main(self.face_frame)

    def main(self, hand, face):
        self._main_hand_rec(hand)
        self._main_face_rec(face)
        self._cv2_show()
        self._send_data()


if __name__ == "__main__":
    main = Main()
    while True:

        _, face_frame = FACE_CAP.read()
        _, hand_frame = HAND_CAP.read()
        hand_frame = cv2.flip(hand_frame, -1)

        main.main(hand_frame, face_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cv2.destroyAllWindows()
            sys.exit(0)
