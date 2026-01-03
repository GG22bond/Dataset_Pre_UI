import json
import os
import cv2
from collections import defaultdict
import argparse
from PySide6.QtCore import QFile, Qt, QCoreApplication

def coco_to_labelme_json(json_path, img_path, update_callback):

    with open(json_path, encoding='utf-8') as f:
        annotation_json = json.load(f)

    if 'categories' in annotation_json:
        cat_map = {cat['id']: cat['name'] for cat in annotation_json['categories']}
    else:
        cat_map = {}

    anns_per_image = defaultdict(list)
    for ann in annotation_json['annotations']:
        anns_per_image[ann['image_id']].append(ann)

    labels = {}

    for img_info in annotation_json['images']:
        img_id = img_info['id']
        image_name = img_info['file_name']
        image_path = os.path.join(img_path, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"[Error] 无法读取图像: {image_path}")
            continue

        h, w = image.shape[:2]

        labelme_dict = {
            'flags': {},
            'imagePath': os.path.basename(image_path),
            'shapes': [],
            'imageData': None,
            'imageHeight': h,
            'imageWidth': w
        }

        for ann in anns_per_image[img_id]:
            cid = ann['category_id']

            cname = ann.get('category_name', cat_map.get(cid, str(cid)))
            labels[cid] = cname

            x, y, bw, bh = ann['bbox']
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + bw), int(y + bh)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

            shape_dict = {
                "label": cname,
                "shape_type": "polygon",
                "category_id": cid,
                "points": []
            }

            segs = ann.get('segmentation', None)

            if isinstance(segs, list) and len(segs) > 0 and isinstance(segs[0], list) and len(segs[0]) >= 6:
                pts = segs[0]
                for idx in range(0, len(pts), 2):
                    px, py = int(pts[idx]), int(pts[idx + 1])
                    cv2.circle(image, (px, py), 3, (0, 0, 213), -1)
                    shape_dict["points"].append([px, py])
            else:

                shape_dict["points"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

            labelme_dict['shapes'].append(shape_dict)

        json_save_path = os.path.splitext(image_path)[0] + ".json"
        with open(json_save_path, 'w', encoding='utf-8') as wf:
            wf.write(json.dumps(labelme_dict, indent=2, ensure_ascii=False))

        # print(f"{os.path.basename(json_path)} convert to {os.path.basename(json_save_path)}")
        message1 = f"{os.path.basename(json_path)} convert to {os.path.basename(json_save_path)}"
        update_callback(message1)
        QCoreApplication.processEvents()


    sorted_cids = sorted(labels.keys())
    sorted_names = [labels[cid] for cid in sorted_cids]
    # print("====== Categories ======")
    # print(sorted_names)
    message2 = f"====== Categories ======\n{sorted_names}"
    update_callback(message2)
    QCoreApplication.processEvents()


# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--json_dir', type=str, default=r'F:\COCO128\val\annotations\instances_val2017.json')
#     parser.add_argument('--img_path', type=str, default=r'F:\COCO128\val\val2017')
#
#     args = parser.parse_args()
#
#     coco_to_labelme_json(args.json_dir, args.img_path)

