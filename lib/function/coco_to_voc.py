import os
import json
import cv2
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from PySide6.QtCore import QFile, Qt, QCoreApplication

def indent_xml(elem, level=0):

    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent_xml(child, level+1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

def create_voc_xml(image_info, anns, category_map, images_dir=None):

    file_name = image_info["file_name"]
    img_width = image_info["width"]
    img_height = image_info["height"]
    img_id = image_info["id"]

    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    if images_dir:
        folder.text = os.path.basename(images_dir.rstrip("/\\"))
    else:
        folder.text = ""


    filename_tag = ET.SubElement(annotation, "filename")
    filename_tag.text = file_name

    path_tag = ET.SubElement(annotation, "path")
    if images_dir:
        path_tag.text = os.path.join(images_dir, file_name)
    else:
        path_tag.text = ""

    # <source><database>Unknown</database></source>
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    # <size><width>..</width><height>..</height><depth> </depth></size>
    size = ET.SubElement(annotation, "size")
    width_tag = ET.SubElement(size, "width")
    width_tag.text = str(img_width)
    height_tag = ET.SubElement(size, "height")
    height_tag.text = str(img_height)

    depth_value = 3
    if images_dir:
        img_path = os.path.join(images_dir, file_name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is not None:

            if len(img.shape) == 2:
                depth_value = 1
            else:
                depth_value = img.shape[2]

    depth_tag = ET.SubElement(size, "depth")
    depth_tag.text = str(depth_value)

    # <segmented>0</segmented>
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    for ann in anns:
        category_id = ann["category_id"]
        bbox = ann["bbox"]
        x_min = int(bbox[0])
        y_min = int(bbox[1])
        x_max = int(bbox[0] + bbox[2])
        y_max = int(bbox[1] + bbox[3])

        # <object>
        obj = ET.SubElement(annotation, "object")
        #   <name>category_name</name>
        name = ET.SubElement(obj, "name")
        name.text = category_map[category_id]
        #   <pose>Unspecified</pose>
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        #   <truncated>0</truncated>
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        #   <difficult>0</difficult>
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        #   <bndbox>
        bndbox = ET.SubElement(obj, "bndbox")
        xmin_tag = ET.SubElement(bndbox, "xmin")
        xmin_tag.text = str(x_min)
        ymin_tag = ET.SubElement(bndbox, "ymin")
        ymin_tag.text = str(y_min)
        xmax_tag = ET.SubElement(bndbox, "xmax")
        xmax_tag.text = str(x_max)
        ymax_tag = ET.SubElement(bndbox, "ymax")
        ymax_tag.text = str(y_max)

    # XML缩进处理
    indent_xml(annotation)
    return annotation

def convert_coco_to_voc(coco_json_file, save_path, update_callback, images_dir=None):

    with open(coco_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    category_map = {}
    for cat in data.get("categories", []):
        cat_id = cat["id"]
        cat_name = cat["name"]
        category_map[cat_id] = cat_name
        # print(f"id: {cat_id}, name: {cat_name}")

        message1 = f"id: {cat_id}, name: {cat_name}"
        update_callback(message1)
        QCoreApplication.processEvents()


    imgid2anns = defaultdict(list)
    for ann in data.get("annotations", []):
        imgid2anns[ann["image_id"]].append(ann)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for img in data.get("images", []):
        img_id = img["id"]
        file_name = img["file_name"]
        xml_tree = create_voc_xml(img, imgid2anns.get(img_id, []), category_map, images_dir)
        xml_filename = os.path.splitext(file_name)[0] + ".xml"
        xml_path = os.path.join(save_path, xml_filename)
        tree = ET.ElementTree(xml_tree)
        tree.write(xml_path, encoding="utf-8", xml_declaration=False)

        # print(f"{os.path.basename(coco_json_file)} saved to: {xml_path}")
        message2 = f"{os.path.basename(coco_json_file)} saved to: {xml_path}"
        update_callback(message2)
        QCoreApplication.processEvents()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default=r"F:\COCO128\val\annotations\instances_val2017.json", help="输入的 COCO 格式 JSON 文件路径")
    parser.add_argument("--save_path", type=str, default=r"F:\COCO128\val\VOC\val", help="输出的 VOC XML 文件存放目录")
    parser.add_argument("--images_dir", type=str, default=r"F:\COCO128\val\val2017", help="（可选）图片所在的文件夹路径，用于在 <path> 中写入完整路径。若不需要可不传。")
    args = parser.parse_args()

    def call_back(m):
        return

    convert_coco_to_voc(coco_json_file=args.json_path,
                        save_path=args.save_path,
                        update_callback=call_back,
                        images_dir=args.images_dir)

