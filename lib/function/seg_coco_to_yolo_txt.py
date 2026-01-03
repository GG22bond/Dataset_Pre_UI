import os
import json
import argparse
from PySide6.QtCore import QFile, Qt, QCoreApplication

def write_yolo_txt_file(txt_file_path, label_seg_x_y_list):
    if not os.path.exists(txt_file_path):
        with open(txt_file_path, "w", encoding='utf-8') as file:
            for element in label_seg_x_y_list:
                file.write(str(element) + " ")
            file.write('\n')
    else:
        with open(txt_file_path, "a", encoding='utf-8') as file:
            for element in label_seg_x_y_list:
                file.write(str(element) + " ")
            file.write('\n')

def coco_to_yolo_txt(in_json_path,  target_dir, update_callback):

    os.makedirs(target_dir, exist_ok=True)

    with open(in_json_path, "r", encoding='utf-8') as f:
        json_data = json.load(f)

    categories = json_data.get('categories', [])

    categories_sorted = sorted(categories, key=lambda x: x['id'])

    id_map = {cat['id']: idx for idx, cat in enumerate(categories_sorted)}

    image_dict = {img['id']: img for img in json_data.get('images', [])}


    for annotation in json_data.get('annotations', []):

        orig_cat_id = annotation.get('category_id')
        image_id    = annotation.get('image_id')

        if orig_cat_id not in id_map:
            continue
        new_cat_id = id_map[orig_cat_id]

        segmentation = annotation.get('segmentation')

        if not isinstance(segmentation, list) or len(segmentation) == 0:
            continue

        poly = segmentation[0]
        if not isinstance(poly, list) or len(poly) < 6:
            continue

        img_info = image_dict.get(image_id)
        if img_info is None:
            continue

        width  = img_info.get('width')
        height = img_info.get('height')

        img_file_name = img_info.get('file_name')
        if img_file_name is None:

            coco_url = img_info.get('coco_url', "")
            img_file_name = coco_url.split("/")[-1] if "/" in coco_url else None
        if not img_file_name:
            continue

        base_name = os.path.splitext(img_file_name)[0]
        txt_file_name = base_name + ".txt"

        try:
            seg_x_y_list = [
                coord / width if idx % 2 == 0 else coord / height
                for idx, coord in enumerate(poly)
            ]
        except Exception:
            continue

        label_seg_x_y_list = seg_x_y_list[:]
        label_seg_x_y_list.insert(0, new_cat_id)

        txt_file_path = os.path.join(target_dir, txt_file_name)
        write_yolo_txt_file(txt_file_path, label_seg_x_y_list)

        print(f"{os.path.basename(in_json_path)} converted to {txt_file_name}")
        message = f"{os.path.basename(in_json_path)} converted to {txt_file_name}"
        update_callback(message)
        QCoreApplication.processEvents()


# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--json_dir', type=str, default=r"F:\COCO128\val\annotations\instances_val2017.json", help="COCO 格式的 JSON 文件")
#     parser.add_argument('--out_dir', type=str, default=r"F:\COCO128\val\yolo", help="输出的 YOLO 格式标注目录")
#     args = parser.parse_args()
#
#     read_json(args.json_dir, args.out_dir)

