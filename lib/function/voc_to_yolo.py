import os
import xml.etree.ElementTree as ET

from PySide6.QtCore import QFile, Qt, QCoreApplication

def convert(size, box):

    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = ((box[0] + box[1]) / 2.0 - 1) * dw
    y = ((box[2] + box[3]) / 2.0 - 1) * dh
    w = (box[1] - box[0]) * dw
    h = (box[3] - box[2]) * dh
    return (x, y, w, h)


def convert_annotation(xml_dir, txt_dir, classes, update_callback):

    os.makedirs(txt_dir, exist_ok=True)

    for fname in os.listdir(xml_dir):
        if not fname.lower().endswith('.xml'):
            print(f"Skipping non-XML file: {fname}")
            continue

        xml_path = os.path.join(xml_dir, fname)
        txt_name = os.path.splitext(fname)[0] + '.txt'
        txt_path = os.path.join(txt_dir, txt_name)

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            # 读取图像宽高
            size_node = root.find('size')
            w = float(size_node.find('width').text)
            h = float(size_node.find('height').text)

            lines = []
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in classes:
                    classes.append(cls)
                cls_id = classes.index(cls)

                xmlbox = obj.find('bndbox')
                xmin = float(xmlbox.find('xmin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymin = float(xmlbox.find('ymin').text)
                ymax = float(xmlbox.find('ymax').text)

                x_center, y_center, bw, bh = convert((w, h), (xmin, xmax, ymin, ymax))
                line = f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
                lines.append(line)

            if lines:
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines))
            message1 = f"Converted successfully: {fname} -> {txt_name}"
            # print(f"Converted successfully: {fname} -> {txt_name}")
        except Exception as e:
            message1 = f"Conversion failed: {fname}\n  Error: {e}"
            # print(f"Conversion failed: {fname}\n  Error: {e}")
        update_callback(message1)
        QCoreApplication.processEvents()

    # print("All files converted successfully.")

    # 保存最终类别列表
    classes_file = os.path.join(txt_dir, 'classes.txt')
    with open(classes_file, 'w', encoding='utf-8') as cf:
        for cls in classes:
            cf.write(cls + '\n')
    message2 = f"\nClasses converted successfully: {classes_file}"
    # print(f"Classes list saved to: {classes_file}")

    update_callback(message2)
    QCoreApplication.processEvents()