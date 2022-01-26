#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import cv2


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
        self.hand_wide = int(self.y_max_max_point[0] + self.y_max_min_point[0])

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
            app_value_x = int(palm_point[0] - (1.2 * max_distance * (palm_point[0] - hand_wide_middle_point[0])))
            app_value_y = int(palm_point[1] + (1.2 * max_distance * abs(palm_point[1] - hand_wide_middle_point[1])))
            wrist_point = (app_value_x / distance_middle, app_value_y / distance_middle)
            self.draw_point_list.append(wrist_point)
        except ZeroDivisionError:
            raise ValueError  # todo 修改错误类型
        return wrist_point

    def _palm_calculate(self):
        """
        计算手心的位置
        :return: 手心所在的坐标 以及 圆的半径
        """
        # if self.y_max[0][1] < img.shape[0] - 10 or self.hand_wide <= 20
        if self.hand_wide <= 20:
            self.hand_wide = 60
            near_point = (int((self.y_max[0][0] + self.y_min[0][0]) / 2),
                          int((self.y_max[0][1] + self.y_min[0][1]) / 2))
            if int(abs(self.y_max[0][1] - self.y_min[0][1])) <= 210:
                raise ValueError("无法计算手心。袖子遮挡")
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
            int(near_point[0] - self.hand_wide / 2), near_point[1])
        zone_point_max = (
            int(near_point[0] + self.hand_wide / 2), near_point[1] + self.hand_wide)
        max_distance = 0
        palm_point = (-1, -1)
        for i in range(zone_point_min[0], zone_point_max[0], 2):
            for j in range(zone_point_min[1], zone_point_max[1], 2):
                distance = cv2.pointPolygonTest(self.contour, (i, j), True)
                if distance > max_distance:
                    max_distance = int(distance)
                    palm_point = (i, j)
        self.draw_point_list.append(palm_point)
        return palm_point, max_distance

    def main_calculate(self, contour):
        """
        主接口，返回各个点的数据
        :except ValueError 袖子遮挡
        :param contour: 手的轮廓
        :return: 返回包含基础数据的字典
        """
        self._load_contour_data(contour)
        palm_point, max_distance = self._palm_calculate()
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
        max_min = np.eval(min_or_max)(contour, axis=0)
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
