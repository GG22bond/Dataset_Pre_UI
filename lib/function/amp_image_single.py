import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
import xml.etree.ElementTree as ET
import argparse

class Single_DataAugmentForObjectDetection:
    def __init__(self,
                params_list2,
                 is_addNoise=True, is_changeLight=True,
                 is_cutout=True, is_rotate_img_bbox=True, is_shift_pic_bboxes=True, is_flip_pic_bboxes=True):

        (self.variance, self.alpha_value, self.cut_out_length,
         self.cut_out_holes, self.cut_out_threshold, self.max_rotation_angle) = params_list2

        # 是否使用对应增强方法
        self.is_addNoise = is_addNoise
        self.is_changeLight = is_changeLight
        self.is_cutout = is_cutout
        self.is_rotate_img_bbox = is_rotate_img_bbox
        self.is_shift_pic_bboxes = is_shift_pic_bboxes
        self.is_flip_pic_bboxes = is_flip_pic_bboxes

    # ----1. 加噪声---- #
    def add_noise(self, img):
        noisy = random_noise(img, mode="gaussian", var=self.variance, clip=True) * 255
        return noisy.astype(img.dtype)

    # ---2. 调整亮度--- #
    def change_light(self, img):
        blank = np.zeros(img.shape, img.dtype)
        adjusted = cv2.addWeighted(img, self.alpha_value, blank, 1 - self.alpha_value, 0)
        return adjusted

    # ---3. cutout--- #
    def cutout(self, img, bboxes):
        def cal_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            if xB <= xA or yB <= yA:
                return 0.0
            interArea = (xB - xA + 1) * (yB - yA + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            return interArea / float(boxBArea)

        h, w = img.shape[:2]
        mask = np.ones((h, w, 1), np.float32)
        for _ in range(self.cut_out_holes):
            chongdie = True
            while chongdie:
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(y - self.cut_out_length // 2, 0, h)
                y2 = np.clip(y + self.cut_out_length // 2, 0, h)
                x1 = np.clip(x - self.cut_out_length // 2, 0, w)
                x2 = np.clip(x + self.cut_out_length // 2, 0, w)
                chongdie = False
                for box in bboxes:
                    if cal_iou([x1, y1, x2, y2], box) > self.cut_out_threshold:
                        chongdie = True
                        break
            mask[y1:y2, x1:x2, :] = 0.0
        cut_img = img.copy()
        cut_img = cut_img * mask
        return cut_img

    # ---4. 旋转--- #
    def rotate_img_bbox(self, img, bboxes):
        angle = random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
        scale = 1.0
        h, w = img.shape[:2]
        rangle = np.deg2rad(angle)
        nw = int((abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale)
        nh = int((abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale)
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        rot_img = cv2.warpAffine(img, rot_mat, (nw, nh), flags=cv2.INTER_LANCZOS4)
        rot_bboxes = []
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            points = [
                np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1])),
                np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1])),
                np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1])),
                np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
            ]
            concat = np.vstack(points).astype(np.int32)
            rx, ry, rw_box, rh_box = cv2.boundingRect(concat)
            rot_bboxes.append([rx, ry, rx + rw_box, ry + rh_box])
        return rot_img, rot_bboxes

    # ---5. 平移--- #
    def shift_pic_bboxes(self, img, bboxes):
        h, w = img.shape[:2]
        x_min, y_min, x_max, y_max = w, h, 0, 0
        for bbox in bboxes:
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[2])
            y_max = max(y_max, bbox[3])
        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max
        x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)
        M = np.float32([[1, 0, x], [0, 1, y]])
        shift_img = cv2.warpAffine(img, M, (w, h))
        shift_bboxes = []
        for bbox in bboxes:
            shift_bboxes.append([
                bbox[0] + x,
                bbox[1] + y,
                bbox[2] + x,
                bbox[3] + y
            ])
        return shift_img, shift_bboxes

    # ---6. 镜像--- #
    def flip_pic_bboxes(self, img, bboxes):
        h, w = img.shape[:2]
        sed = random.random()
        if sed < 0.33:
            flip_img = cv2.flip(img, 0)
            mode = 0
        elif sed < 0.66:
            flip_img = cv2.flip(img, 1)
            mode = 1
        else:
            flip_img = cv2.flip(img, -1)
            mode = -1

        flip_bboxes = []
        for box in bboxes:
            x_min, y_min, x_max, y_max = box
            if mode == 0:  # 垂直
                flip_bboxes.append([x_min, h - y_max, x_max, h - y_min])
            elif mode == 1:  # 水平
                flip_bboxes.append([w - x_max, y_min, w - x_min, y_max])
            else:  # 对角
                flip_bboxes.append([w - x_max, h - y_max, w - x_min, h - y_min])
        return flip_img, flip_bboxes


class Single_ToolHelper:
    def parse_xml(self, path):

        tree = ET.parse(path)
        root = tree.getroot()
        objs = root.findall('object')
        coords = []
        for obj in objs:
            name = obj.find('name').text
            box = obj.find('bndbox')
            x_min = int(box.find('xmin').text)
            y_min = int(box.find('ymin').text)
            x_max = int(box.find('xmax').text)
            y_max = int(box.find('ymax').text)
            coords.append([x_min, y_min, x_max, y_max, name])
        return coords

    def save_img(self, file_name, save_folder, img):

        cv2.imwrite(os.path.join(save_folder, file_name), img)

    def process_single_image(self, dataAug, image_path, xml_path, save_img_path):

        os.makedirs(save_img_path, exist_ok=True)

        values = self.parse_xml(xml_path)
        coords = [v[:4] for v in values]

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        file_name = os.path.basename(image_path)
        dot_index = file_name.rfind('.')
        prefix = file_name[:dot_index]
        suffix = file_name[dot_index:]

        # 依次做各类增强并保存图像
        if dataAug.is_addNoise:
            noisy_img = dataAug.add_noise(img)
            name_img = f"{prefix}_noise{suffix}"
            self.save_img(name_img, save_img_path, noisy_img)
            print(f"Saved noise image: {name_img}")

        if dataAug.is_changeLight:
            light_img = dataAug.change_light(img)
            name_img = f"{prefix}_light{suffix}"
            self.save_img(name_img, save_img_path, light_img)
            print(f"Saved bright image: {name_img}")

        if dataAug.is_cutout:
            cut_img = dataAug.cutout(img, coords)
            name_img = f"{prefix}_cutout{suffix}"
            self.save_img(name_img, save_img_path, cut_img)
            print(f"Saved cutout image: {name_img}")

        if dataAug.is_rotate_img_bbox:
            rot_img, _ = dataAug.rotate_img_bbox(img, coords)
            name_img = f"{prefix}_rotate{suffix}"
            self.save_img(name_img, save_img_path, rot_img)
            print(f"Saved rotated image: {name_img}")

        if dataAug.is_shift_pic_bboxes:
            shift_img, _ = dataAug.shift_pic_bboxes(img, coords)
            name_img = f"{prefix}_shift{suffix}"
            self.save_img(name_img, save_img_path, shift_img)
            print(f"Saved shifted image: {name_img}")

        if dataAug.is_flip_pic_bboxes:
            flip_img, _ = dataAug.flip_pic_bboxes(img, coords)
            name_img = f"{prefix}_flip{suffix}"
            self.save_img(name_img, save_img_path, flip_img)
            print(f"Saved flipped image: {name_img}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='../test/amp_test/images/4.jpg', help='单张原始图像路径')
    parser.add_argument('--xml_path', type=str, default='../test/amp_test/annotations/4.xml', help='对应的原始 XML 标注路径')
    parser.add_argument('--save_img_path', type=str, default='../test/amp_test/image_single', help='增强后图像保存目录')
    args = parser.parse_args()

    # 设置增强参数：[variance, alpha_value, cut_out_length, cut_out_holes, cut_out_threshold, max_rotation_angle]
    params = [0.01, 0.6, 30, 1, 0.3, 5]

    dataAug = Single_DataAugmentForObjectDetection(params_list2=params)
    toolhelper = Single_ToolHelper()

    toolhelper.process_single_image(dataAug=dataAug, image_path=args.image_path, xml_path=args.xml_path, save_img_path=args.save_img_path)
