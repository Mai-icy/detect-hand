#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2


class WristError(ValueError):
    """手腕被袖子遮挡"""


class CalculateError(ValueError):
    """计算点无法计算，提高下界"""


class CalculationContour:
    def __init__(self):
        self.draw_point_list = []

    def _load_contour_data(self, contour):
        """
        载入contour数据，以及分析基本数据，以供其他函数(_wrist_calculate，_palm_calculate)分析
        :param contour: 轮廓数据
        """
        self.draw_point_list.clear()
        self.contour = contour
        squeeze_contour = np.squeeze(contour)
        self.x_max = x_y_max_min(squeeze_contour, 'max', 'x')
        self.y_max = x_y_max_min(squeeze_contour, 'max', 'y')
        self.x_min = x_y_max_min(squeeze_contour, 'min', 'x')
        self.y_min = x_y_max_min(squeeze_contour, 'min', 'y')
        sort_y_max = sorted(self.y_max)
        self.y_max_min_point = sort_y_max[0]  # 获取最下边的的左和右点
        self.y_max_max_point = sort_y_max[-1]
        self.hand_wide = int(self.y_max_max_point[0] - self.y_max_min_point[0])

        self.draw_point_list.extend([self.x_max[0], self.x_min[0], self.y_min[0],
                                     self.y_max_min_point, self.y_max_max_point])

    def _wrist_calculate(self, palm_point, max_distance):
        """
        计算手腕测温点的位置
        :param palm_point: 手心坐标点
        :param max_distance: 手心圆的半径
        :return: 结果测温点的坐标
        """
        hand_wide_middle_point = (
            int((self.y_max_max_point[0] + self.y_max_min_point[0]) / 2), self.y_max[0][1])
        distance_middle = round(
            ((palm_point[0] - hand_wide_middle_point[0]) ** 2 + (
                palm_point[1] - hand_wide_middle_point[1]) ** 2) ** 0.5)

        if distance_middle == 0:
            distance_middle = 1
        try:
            app_value_x = int((1.2 * max_distance * (palm_point[0] - hand_wide_middle_point[0])) / distance_middle)
            app_value_y = int((1.2 * max_distance * abs(palm_point[1] - hand_wide_middle_point[1])) / distance_middle)
            wrist_point = (palm_point[0] - app_value_x, palm_point[1] + app_value_y)
        except ZeroDivisionError:
            raise CalculateError
        return wrist_point

    def _palm_calculate(self, shape_x):
        """
        计算手心的位置
        :return: 手心所在的坐标 以及 圆的半径
        """
        # if self.y_max[0][1] < img.shape[0] - 10 or self.hand_wide <= 20
        if self.hand_wide <= 35 or self.y_max[0][1] < shape_x - 10:
            raise CalculateError
        elif abs(self.x_min[0][1] - self.y_max_min_point[1]) < self.hand_wide / 8:  # 手倾斜分类讨论
            near_point = (int((self.y_min[0][0] + self.x_max[0][0]) / 2) - self.hand_wide, int(
                (self.y_min[0][1] + self.x_max[0][1]) / 2) + int(self.hand_wide / 2))
        elif abs(self.x_max[0][1] - self.y_max_max_point[1]) < self.hand_wide / 8:
            near_point = (int((self.y_min[0][0] + self.x_min[0][0]) / 2) + self.hand_wide, int(
                (self.y_min[0][1] + self.x_min[0][1]) / 2) + int(self.hand_wide / 2))
        else:
            near_point = (int((self.x_max[0][0] + self.x_min[0][0]) / 2),
                          int((self.x_min[0][1] + self.x_max[0][1]) / 2))
        zone_point_min = (
            int(near_point[0] - self.hand_wide / 1.5), int(near_point[1] - self.hand_wide * 0.5))
        zone_point_max = (
            int(near_point[0] + self.hand_wide / 1.5), int(near_point[1] + self.hand_wide * 1.5))
        max_distance = 0
        palm_point = (-1, -1)
        for i in range(zone_point_min[0], zone_point_max[0], 3):
            for j in range(zone_point_min[1], zone_point_max[1], 3):
                distance = cv2.pointPolygonTest(self.contour, (i, j), True)
                if distance > max_distance:
                    max_distance = int(distance)
                    palm_point = (i, j)
        # self.th_time =
        if palm_point[1] + max_distance * 1.5 >= shape_x:
            raise WristError
        self.draw_point_list.append(palm_point)
        return palm_point, max_distance

    def main_calculate(self, contour, shape):
        """
        主接口，返回各个点的数据

        :param shape: 图片高度
        :except ValueError 袖子遮挡
        :param contour: 手的轮廓
        :return: 返回包含基础数据的字典
        """
        self._load_contour_data(contour)
        palm_point, max_distance = self._palm_calculate(shape)
        wrist_point = self._wrist_calculate(palm_point, max_distance)
        return {
            "palm_point": palm_point,
            "max_distance": max_distance,
            "wrist_point": wrist_point,
            "draw_point_list": self.draw_point_list
        }


def x_y_max_min(contour, min_or_max, x_or_y):
    """
    获取counter数组内x或y的最大或最小值的坐标点

    :param contour: 轮廓数组
    :param min_or_max: 'max' or 'min'
    :param x_or_y: 'x' or 'y'
    :return: 返回坐标点
    """
    if min_or_max == 'max' or min_or_max == 'min':
        max_min = np.max(contour, axis=0) if min_or_max == 'max' else np.min(contour, axis=0)
    else:
        raise ValueError("min_or_max must be 'max' or 'min'")
    if x_or_y == 'x' or x_or_y == 'y':
        line_num = 0 if x_or_y == 'x' else 1
    else:
        raise ValueError("x_or_y must be 'x' or 'y'")
    whe = np.where(contour == max_min[line_num])
    np_list = [whe[0][i]
               for i in range(0, whe[0].shape[0]) if whe[1][i] == line_num]
    result_list = [tuple(contour[x]) for x in np_list]
    return result_list
