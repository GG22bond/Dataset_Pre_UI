import os
import cv2
import json

from tqdm import tqdm
import argparse
from PySide6.QtCore import QFile, Qt, QCoreApplication

'''
    From: https://github.com/z1069614715/objectdetection_script/blob/master/mmdet-course/yolo2coco.py Thank you! 
'''

def convert_yolo_to_coco(images_dir, labels_dir, output_path, classes_list, update_callback):

    # print("Images directory:", images_dir)
    # print("Labels directory:", labels_dir)
    # print("Output path:", output_path)

    # 检查目录
    assert os.path.exists(images_dir), f"Images directory {images_dir} not found"
    assert os.path.exists(labels_dir), f"Labels directory {labels_dir} not found"

    # 收集所有图片文件
    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # 初始化 COCO 数据结构
    coco = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    # 填充 categories
    for i, cls in enumerate(classes_list):
        coco['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

    ann_id = 0
    # 遍历图片并转换标注
    # for img_file in tqdm(image_files, desc="Processing images"):
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        image_id = os.path.splitext(img_file)[0]

        # 添加 image 信息
        coco['images'].append({
            'file_name': img_file,
            'id': image_id,
            'width': w,
            'height': h
        })

        # 读取对应 YOLO 标注
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        if not os.path.exists(label_path):
            continue
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, x_center, y_center, bw, bh = parts
                cls_id = int(cls_id)
                x_center, y_center, bw, bh = map(float, (x_center, y_center, bw, bh))
                x1 = (x_center - bw / 2) * w
                y1 = (y_center - bh / 2) * h
                bbox_w = bw * w
                bbox_h = bh * h
                # 添加 annotation
                coco['annotations'].append({
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': cls_id,
                    'bbox': [x1, y1, bbox_w, bbox_h],
                    'area': bbox_w * bbox_h,
                    'iscrowd': 0,   # coco数据集中的iscrowd: 0 表示这个标注对应的是独立可分的实例, 1 表示这个标注对应的是一大群、堆在一起、无法逐一分割的“人群/物体集合”
                    'segmentation': [[
                        x1, y1,
                        x1 + bbox_w, y1,
                        x1 + bbox_w, y1 + bbox_h,
                        x1, y1 + bbox_h
                    ]]
                })
                ann_id += 1

        # print(f"{image_id} converted to {os.path.basename(output_path)}")
        message1 = f"{image_id} converted to {os.path.basename(output_path)}"
        update_callback(message1)
        QCoreApplication.processEvents()



    # 保存为 COCO JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f)

    # print(f"Saved JSON file {output_path} successfully")
    message2 = f"Saved COCO JSON to {output_path} successfully."
    update_callback(message2)
    QCoreApplication.processEvents()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default= r'F:\Fin_DT\Fin_DT\BOLT\bolt\images\val', help='输入图像文件夹路径')
    parser.add_argument('--labels_dir', type=str, default= r'F:\Fin_DT\Fin_DT\BOLT\bolt\labels\val', help='输入标签文件夹路径')
    parser.add_argument('--output', type=str, default= r'F:\Fin_DT\Fin_DT\BOLT\bolt\coco\val\valid1.json', help='输出 COCO JSON 文件路径和文件名')

    args = parser.parse_args()

    classes = ['HHB', 'HNB', 'FPB']

    def update_callback(msg):
        return

    convert_yolo_to_coco(images_dir=args.images_dir,
                         labels_dir=args.labels_dir,
                         output_path=args.output,
                         classes_list=classes,
                         update_callback=update_callback)

