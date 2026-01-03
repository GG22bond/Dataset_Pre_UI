import os
import json
from tqdm import tqdm
import argparse
from PySide6.QtCore import QFile, Qt, QCoreApplication

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    return (x, y, w, h)

def convert_coco_to_yolo(coco_json_file, save_path, update_callback):

    json_file = coco_json_file
    ana_txt_save_path = save_path

    data = json.load(open(json_file, 'r'))

    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)

    # category['id'] 从小到大排序
    sorted_categories = sorted(data['categories'], key=lambda x: x['id'])

    id_map = {}
    with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
        # 写入classes.txt
        for i, category in enumerate(sorted_categories):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i
    # print(f"Number of categories: {len(sorted_categories)}")
    message1 = f"Number of categories: {len(sorted_categories)}"
    update_callback(message1)
    QCoreApplication.processEvents()

    category_names = [cat['name'] for cat in sorted_categories]
    # print(f"categories: {', '.join(category_names)}")
    message2 = f"categories: {', '.join(category_names)}"
    update_callback(message2)
    QCoreApplication.processEvents()

    # for img in tqdm(data['images']):
    for img in data['images']:
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, _ = os.path.splitext(filename)
        ana_txt_name = head + ".txt"
        try:
            with open(os.path.join(ana_txt_save_path, ana_txt_name), 'w') as f_txt:
                for ann in data['annotations']:
                    if ann['image_id'] == img_id:
                        box = convert((img_width, img_height), ann["bbox"])
                        f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
                        # print(f"Convert to {ana_txt_name}")
                        message3 = f"Convert to {ana_txt_name}"

        except Exception as e:
            # print(f"[Error] image_id={img_id}, file={filename}: {e}")
            message3 = f"[Error] image_id={img_id}, file={filename}: {e}"

        update_callback(message3)
        QCoreApplication.processEvents()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', default=r'F:\COCO128\val\annotations\instances_val2017.json', type=str,
                        help="input: coco format(json)")
    parser.add_argument('--save_path', default=r'F:\COCO128\val\yolo\val', type=str,
                        help="specify where to save the output dir of labels")
    arg = parser.parse_args()

    def update_callback(msg):
        return

    convert_coco_to_yolo(coco_json_file=arg.json_path, save_path=arg.save_path, update_callback=update_callback)

