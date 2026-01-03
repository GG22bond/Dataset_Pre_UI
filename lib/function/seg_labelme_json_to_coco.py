import os
import json
import argparse
import cv2
from PySide6.QtCore import QFile, Qt, QCoreApplication

def polygon_area(xy):

    pts = [(xy[i], xy[i+1]) for i in range(0, len(xy), 2)]
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def labelme_json_to_coco(labelme_dir, img_dir, output_json, update_callback):

    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    catid_to_label = {}

    label_to_catid = {}

    used_cat_ids = set()

    next_ann_id = 1
    next_image_id = 1

    for fname in os.listdir(labelme_dir):
        if not fname.lower().endswith(".json"):
            continue

        labelme_path = os.path.join(labelme_dir, fname)
        with open(labelme_path, "r", encoding="utf-8") as f:
            lm = json.load(f)

        image_name = lm.get("imagePath", None)
        if image_name is None:
            print(f"[Warning] `{labelme_path}` 中缺少 'imagePath'，跳过该文件。")
            continue

        img_fullpath = os.path.join(img_dir, image_name)
        img_height = lm.get("imageHeight", None)
        img_width  = lm.get("imageWidth", None)
        if img_height is None or img_width is None:
            img = cv2.imread(img_fullpath)
            if img is None:
                print(f"[Error] 无法读取图像 `{img_fullpath}`，跳过。")
                continue
            img_height, img_width = img.shape[:2]
        else:
            if not os.path.exists(img_fullpath):
                print(f"[Error] 图像 `{img_fullpath}` 不存在，跳过。")
                continue

        image_id = next_image_id
        next_image_id += 1

        image_info = {
            "id": image_id,
            "file_name": image_name,
            "height": img_height,
            "width": img_width
        }
        coco_dict["images"].append(image_info)

        for shape in lm.get("shapes", []):
            label = shape.get("label", None)
            if label is None:
                continue

            if "category_id" in shape:
                try:
                    cid = int(shape["category_id"])
                except:
                    print(f"[Warning] 在 `{labelme_path}` 的某个 shape 中，"
                          f"`category_id` 无法转换为整数，跳过该 shape。")
                    continue

                if cid in catid_to_label:

                    if catid_to_label[cid] != label:
                        print(f"[Warning] category_id={cid} 在不同 shape 中对应了 "
                              f"'{catid_to_label[cid]}' 与 '{label}'，使用第一个。")
                else:
                    catid_to_label[cid] = label

                used_cat_ids.add(cid)
                category_id = cid

            else:

                if label in label_to_catid:
                    category_id = label_to_catid[label]
                else:

                    new_id = 1
                    while new_id in used_cat_ids:
                        new_id += 1
                    label_to_catid[label] = new_id
                    catid_to_label[new_id] = label
                    used_cat_ids.add(new_id)
                    category_id = new_id

            pts2d = shape.get("points", [])
            if len(pts2d) < 3:

                continue

            xy = []
            for p in pts2d:
                xy.extend([float(p[0]), float(p[1])])

            x_coords = [p[0] for p in pts2d]
            y_coords = [p[1] for p in pts2d]
            xmin = min(x_coords)
            ymin = min(y_coords)
            xmax = max(x_coords)
            ymax = max(y_coords)
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin

            segmentation = [xy]

            area = polygon_area(xy)

            ann = {
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": segmentation,
                "area": area,
                "bbox": [xmin, ymin, bbox_w, bbox_h],
                "iscrowd": 0
            }
            coco_dict["annotations"].append(ann)
            next_ann_id += 1

        # print(f"{os.path.basename(labelme_path)} converted to {os.path.basename(output_json)}")
        message1 = f"{os.path.basename(labelme_path)} converted to {os.path.basename(output_json)}"
        update_callback(message1)
        QCoreApplication.processEvents()


    for cid in sorted(used_cat_ids):
        cat = {
            "id": cid,
            "name": catid_to_label[cid],
            "supercategory": ""
        }
        coco_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, ensure_ascii=False, separators=(",", ":"))

    print(f"Images number: {len(coco_dict['images'])}")
    print(f"Annotations number: {len(coco_dict['annotations'])}")
    print(f"Categories number: {len(coco_dict['categories'])}")

    message2 = (f"Images number: {len(coco_dict['images'])} \n"
                f"Annotations number: {len(coco_dict['annotations'])} \n"
                f"Categories number: {len(coco_dict['categories'])}")

    update_callback(message2)
    QCoreApplication.processEvents()

# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--labelme_dir", type=str, default=r'F:\COCO128\val\val2017')
#     parser.add_argument("--img_dir", type=str, default=r'F:\COCO128\val\val2017')
#     parser.add_argument("--output_json", type=str, default=r'F:\COCO128\val\annotations\val.json')
#     args = parser.parse_args()
#
#     labelme_json_to_coco(args.labelme_dir, args.img_dir, args.output_json)

