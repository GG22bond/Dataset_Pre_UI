import os
import cv2
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
import argparse
from PySide6.QtCore import QFile, Qt, QCoreApplication

def convert_voc_to_coco(images_dir, annotations_dir, output_path, classes_list, update_callback):

    assert os.path.exists(images_dir), f"Images directory {images_dir} not found"
    assert os.path.exists(annotations_dir), f"Annotations directory {annotations_dir} not found"

    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    coco = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    for i, cls in enumerate(classes_list):
        coco['categories'].append({
            'id': i,
            'name': cls,
            'supercategory': 'mark'
        })
        # print(f"id: {i}, name: {cls}")
        message1 = f"id: {i}, name: {cls}"
        update_callback(message1)
        QCoreApplication.processEvents()


    ann_id = 0

    # for img_file in tqdm(image_files, desc="Processing images"):
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        image_id = os.path.splitext(img_file)[0]

        coco['images'].append({
            'file_name': img_file,
            'id': image_id,
            'width': w,
            'height': h
        })

        xml_file = os.path.splitext(img_file)[0] + '.xml'
        xml_path = os.path.join(annotations_dir, xml_file)
        if not os.path.exists(xml_path):
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in classes_list:
                continue
            category_id = classes_list.index(class_name)

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            bbox_w = xmax - xmin
            bbox_h = ymax - ymin
            x1 = xmin
            y1 = ymin

            coco['annotations'].append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [x1, y1, bbox_w, bbox_h],
                'area': bbox_w * bbox_h,
                'iscrowd': 0,
                'segmentation': [[
                    x1, y1,
                    x1 + bbox_w, y1,
                    x1 + bbox_w, y1 + bbox_h,
                    x1, y1 + bbox_h
                ]]
            })
            ann_id += 1

        # print(f"{image_id} converted to {os.path.basename(output_path)}")
        message2 = f"{image_id} converted to {os.path.basename(output_path)}"
        update_callback(message2)
        QCoreApplication.processEvents()


    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # categories → annotations → images
    ordered_coco = {
        'categories': coco['categories'],
        'annotations': coco['annotations'],
        'images': coco['images']
    }

    with open(output_path, 'w') as f:
        json.dump(ordered_coco, f)

    # print(f"Saved COCO JSON to {output_path} successfully.")
    message3 = f"Saved COCO JSON to {output_path} successfully."
    update_callback(message3)
    QCoreApplication.processEvents()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default=r"E:\mmdetection\data\bolt\valid\images", help='输入图像文件夹路径')
    parser.add_argument('--annotations_dir', type=str, default=r"F:\Fin_DT\Fin_DT\BOLT\bolt\xml\val", help='输入 VOC XML 注释文件夹路径')
    parser.add_argument('--output', type=str, default=r"F:\Fin_DT\Fin_DT\BOLT\bolt\coco\val\val_1.json", help='输出 COCO JSON 文件路径和文件名')

    args = parser.parse_args()

    classes = ['HHB', 'HNB', 'FPB']

    def update_callback(msg):
        return

    convert_voc_to_coco(images_dir=args.images_dir,
             annotations_dir=args.annotations_dir,
             output_path=args.output,
             classes_list=classes,
             update_callback=update_callback)
