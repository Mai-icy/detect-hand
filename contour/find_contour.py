#!/usr/bin/python
# -*- coding:utf-8 -*-
import cv2
import numpy as np


def Interception_contour(res):
    """
    在Ostu处理之后的图片进行手轮廓的获取
    :param res: skin_mask函数返回的图片
    :return: 返回轮廓二维np数组
    """
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    Laplacian = cv2.convertScaleAbs(dst)
    contour = cv2.findContours(Laplacian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    contour = sorted(contour, key=cv2.contourArea, reverse=True)
    return contour[0]


def skin_mask(roi):
    """
    *进行Ostu自动分割处理*
    YCrCb颜色空间的Cr分量+Otsu法阈值分割算法
    :param roi: 原始图片
    :return: 通过Otsu自动分割得到的图片
    """
    YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)  # 转换至YCrCb空间
    (y, cr, cb) = cv2.split(YCrCb)  # 拆分出Y,Cr,Cb值
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)  # 高斯滤波GaussianBlur
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Ostu处理
    res = cv2.bitwise_and(roi, roi, mask=skin)
    return res


def pre_cut_color(img, lower, upper):
    """
    *进行颜色处理*
    抓取手肤色的部分，
    :param img:
    :param lower: YCRCB颜色控件下限值 示例：np.array([70, 138, 86], np.uint8)
    :param upper: YCRCB颜色控件上限值 示例：np.array([168, 160, 123], np.uint8)
    :return: 返回 黑白图片 以及 最大的轮廓数据
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    mask = cv2.inRange(ycrcb, lower, upper)
    _, black_and_white = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(
        black_and_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(black_and_white, contours, -1, (0, 255, 0), 3)
    if len(contours) == 0:  # 防止没有符合条件的轮廓而报错
        biggest_contour = None
    else:
        biggest_contour = sorted(contours, key=cv2.contourArea, reverse=True)
    return black_and_white, biggest_contour
