import math

import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image


#   计算原始输入图像
#   每一次缩放的比例
def calculateScales(img):
    pr_scale = 1.0
    h, w, _ = img.shape
    #   将最大的图像大小进行一个固定
    #   如果图像的短边大于500，则将短边固定为500
    #   如果图像的长边小于500，则将长边固定为500
    if min(w, h) > 500:
        pr_scale = 500.0 / min(h, w)
        w = int(w * pr_scale)
        h = int(h * pr_scale)
    elif max(w, h) < 500:
        pr_scale = 500.0 / max(h, w)
        w = int(w * pr_scale)
        h = int(h * pr_scale)

    #   建立图像金字塔的scales，防止图像的宽高小于12
    scales = []
    factor = 0.709
    factor_count = 0
    minl = min(h, w)
    while minl >= 12:
        scales.append(pr_scale * pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales

#   将长方形调整为正方形


def rect2square(rectangles):
    w = rectangles[:, 2] - rectangles[:, 0]
    h = rectangles[:, 3] - rectangles[:, 1]
    l = np.maximum(w, h).T
    rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - l * 0.5
    rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - l * 0.5
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T
    return rectangles

#   非极大抑制


def NMS(rectangles, threshold):
    if len(rectangles) == 0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    I = np.array(s.argsort())
    pick = []
    while len(I) > 0:
        # I[-1] have hightest prob score, I[0:-1]->others
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o <= threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle


def detect_face_12net(
        cls_prob,
        roi,
        out_side,
        scale,
        width,
        height,
        threshold):
    #   计算特征点之间的步长
    stride = 0
    if out_side != 1:
        stride = float(2 * out_side - 1) / (out_side - 1)

    #   获得满足得分门限的特征点的坐标
    (y, x) = np.where(cls_prob >= threshold)

    #   获得满足得分门限的特征点得分
    #   最终获得的score的shape为：[num_box, 1]
    score = np.expand_dims(cls_prob[y, x], -1)

    #   将对应的特征点的坐标转换成位于原图上的先验框的坐标
    #   利用回归网络的预测结果对先验框的左上角与右下角进行调整
    #   获得对应的粗略预测框
    #   最终获得的boundingbox的shape为：[num_box, 4]
    bounding_box = np.concatenate(
        [np.expand_dims(x, -1), np.expand_dims(y, -1)], axis=-1)
    top_left = np.fix(stride * bounding_box + 0)
    bottom_right = np.fix(stride * bounding_box + 11)
    bounding_box = np.concatenate((top_left, bottom_right), axis=-1)
    bounding_box = (bounding_box + roi[y, x] * 12.0) * scale

    #   将预测框和得分进行堆叠，并转换成正方形
    #   最终获得的rectangles的shape为：[num_box, 5]
    rectangles = np.concatenate((bounding_box, score), axis=-1)
    rectangles = rect2square(rectangles)

    rectangles[:, [1, 3]] = np.clip(rectangles[:, [1, 3]], 0, height)
    rectangles[:, [0, 2]] = np.clip(rectangles[:, [0, 2]], 0, width)
    return rectangles


def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):
    #   利用得分进行筛选
    pick = cls_prob[:, 1] >= threshold

    score = cls_prob[pick, 1:2]
    rectangles = rectangles[pick, :4]
    roi = roi[pick, :]

    #   利用Rnet网络的预测结果对粗略预测框进行调整
    #   最终获得的rectangles的shape为：[num_box, 4]
    w = np.expand_dims(rectangles[:, 2] - rectangles[:, 0], -1)
    h = np.expand_dims(rectangles[:, 3] - rectangles[:, 1], -1)
    rectangles[:, [0, 2]] = rectangles[:, [0, 2]] + roi[:, [0, 2]] * w
    rectangles[:, [1, 3]] = rectangles[:, [1, 3]] + roi[:, [1, 3]] * w

    #   将预测框和得分进行堆叠，并转换成正方形
    #   最终获得的rectangles的shape为：[num_box, 5]
    rectangles = np.concatenate((rectangles, score), axis=-1)
    rectangles = rect2square(rectangles)

    rectangles[:, [1, 3]] = np.clip(rectangles[:, [1, 3]], 0, height)
    rectangles[:, [0, 2]] = np.clip(rectangles[:, [0, 2]], 0, width)
    return np.array(NMS(rectangles, 0.7))


def filter_face_48net(
        cls_prob,
        roi,
        pts,
        rectangles,
        width,
        height,
        threshold):
    #   利用得分进行筛选
    pick = cls_prob[:, 1] >= threshold

    score = cls_prob[pick, 1:2]
    rectangles = rectangles[pick, :4]
    pts = pts[pick, :]
    roi = roi[pick, :]

    w = np.expand_dims(rectangles[:, 2] - rectangles[:, 0], -1)
    h = np.expand_dims(rectangles[:, 3] - rectangles[:, 1], -1)
    #   利用Onet网络的预测结果对预测框进行调整
    #   通过解码获得人脸关键点与预测框的坐标
    #   最终获得的face_marks的shape为：[num_box, 10]
    #   最终获得的rectangles的shape为：[num_box, 4]
    face_marks = np.zeros_like(pts)
    face_marks[:, [0, 2, 4, 6, 8]] = w * \
        pts[:, [0, 1, 2, 3, 4]] + rectangles[:, 0:1]
    face_marks[:, [1, 3, 5, 7, 9]] = h * \
        pts[:, [5, 6, 7, 8, 9]] + rectangles[:, 1:2]
    rectangles[:, [0, 2]] = rectangles[:, [0, 2]] + roi[:, [0, 2]] * w
    rectangles[:, [1, 3]] = rectangles[:, [1, 3]] + roi[:, [1, 3]] * w
    #   将预测框和得分进行堆叠
    #   最终获得的rectangles的shape为：[num_box, 15]
    rectangles = np.concatenate((rectangles, score, face_marks), axis=-1)

    rectangles[:, [1, 3]] = np.clip(rectangles[:, [1, 3]], 0, height)
    rectangles[:, [0, 2]] = np.clip(rectangles[:, [0, 2]], 0, width)
    return np.array(NMS(rectangles, 0.3))

#   人脸对齐


def Alignment_1(img, landmark):

    if landmark.shape[0] == 68:
        x = landmark[36, 0] - landmark[45, 0]
        y = landmark[36, 1] - landmark[45, 1]
    elif landmark.shape[0] == 5:
        x = landmark[0, 0] - landmark[1, 0]
        y = landmark[0, 1] - landmark[1, 1]
    # 眼睛连线相对于水平线的倾斜角
    if x == 0:
        angle = 0
    else:
        # 计算它的弧度制
        angle = math.atan(y / x) * 180 / math.pi

    center = (img.shape[1] // 2, img.shape[0] // 2)

    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    # 仿射函数
    new_img = cv2.warpAffine(img, RotationMatrix, (img.shape[1], img.shape[0]))

    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = [RotationMatrix[0, 0] *
               landmark[i, 0] +
               RotationMatrix[0, 1] *
               landmark[i, 1] +
               RotationMatrix[0, 2], RotationMatrix[1, 0] *
               landmark[i, 0] +
               RotationMatrix[1, 1] *
               landmark[i, 1] +
               RotationMatrix[1, 2]]
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark


def Alignment_2(img, std_landmark, landmark):
    def Transformation(std_landmark, landmark):
        std_landmark = np.matrix(std_landmark).astype(np.float64)
        landmark = np.matrix(landmark).astype(np.float64)

        c1 = np.mean(std_landmark, axis=0)
        c2 = np.mean(landmark, axis=0)
        std_landmark -= c1
        landmark -= c2

        s1 = np.std(std_landmark)
        s2 = np.std(landmark)
        std_landmark /= s1
        landmark /= s2

        U, S, Vt = np.linalg.svd(std_landmark.T * landmark)
        R = (U * Vt).T

        return np.vstack(
            [np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

    Trans_Matrix = Transformation(std_landmark, landmark)  # Shape: 3 * 3
    Trans_Matrix = Trans_Matrix[:2]
    Trans_Matrix = cv2.invertAffineTransform(Trans_Matrix)
    new_img = cv2.warpAffine(img, Trans_Matrix, (img.shape[1], img.shape[0]))

    Trans_Matrix = np.array(Trans_Matrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = [Trans_Matrix[0, 0] *
               landmark[i, 0] +
               Trans_Matrix[0, 1] *
               landmark[i, 1] +
               Trans_Matrix[0, 2], Trans_Matrix[1, 0] *
               landmark[i, 0] +
               Trans_Matrix[1, 1] *
               landmark[i, 1] +
               Trans_Matrix[1, 2]]
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
    h, w = input_shape

    new_ar = w / h * rand(1 - jitter, 1 + jitter) / \
        rand(1 - jitter, 1 + jitter)
    scale = rand(.7, 1.3)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (0, 0, 0))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x) * 255  # numpy array, 0 to 1
    return image_data