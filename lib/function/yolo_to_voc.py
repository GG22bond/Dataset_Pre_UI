import os
import cv2
from xml.dom.minidom import Document
from PySide6.QtCore import QFile, Qt, QCoreApplication


def makexml(picPath, txtPath, xmlPath, classes, update_callback):

    datasetName = 'dataset'
    # logs = []
    files = os.listdir(txtPath)
    os.makedirs(xmlPath, exist_ok=True)
    for name in files:
        if name == 'classes.txt' or not name.endswith('.txt'):
            message0 = f"Filtered files: {name}"
            update_callback(message0)
            QCoreApplication.processEvents()
            continue

        txt_file = os.path.join(txtPath, name)
        xml_name = name.replace('.txt', '.xml')
        xml_file = os.path.join(xmlPath, xml_name)
        try:
            xmlBuilder = Document()
            annotation = xmlBuilder.createElement("annotation")
            xmlBuilder.appendChild(annotation)

            with open(txt_file, 'r', encoding='utf-8') as f:
                txtList = f.readlines()

            img_path = os.path.join(picPath, name.replace('.txt', '.jpg'))
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            Pheight, Pwidth, Pdepth = img.shape

            # folder, filename, size
            folder = xmlBuilder.createElement("folder")
            folder.appendChild(xmlBuilder.createTextNode(datasetName))
            annotation.appendChild(folder)
            filename = xmlBuilder.createElement("filename")
            filename.appendChild(xmlBuilder.createTextNode(name[:-4] + ".jpg"))
            annotation.appendChild(filename)
            size = xmlBuilder.createElement("size")
            for tag, val in [("width", Pwidth), ("height", Pheight), ("depth", Pdepth)]:
                elem = xmlBuilder.createElement(tag)
                elem.appendChild(xmlBuilder.createTextNode(str(val)))
                size.appendChild(elem)
            annotation.appendChild(size)

            # objects
            for line in txtList:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, x_center, y_center, w_norm, h_norm = parts
                try:
                    cls_idx = int(cls_id)
                except ValueError:
                    continue
                if cls_idx < 0 or cls_idx >= len(classes):
                    continue
                obj = xmlBuilder.createElement("object")
                # name, pose, truncated, difficult
                name_tag = xmlBuilder.createElement("name")
                name_tag.appendChild(xmlBuilder.createTextNode(classes[cls_idx]))
                obj.appendChild(name_tag)
                for tag, text in [("pose", "Unspecified"), ("truncated", "0"), ("difficult", "0")]:
                    elem = xmlBuilder.createElement(tag)
                    elem.appendChild(xmlBuilder.createTextNode(text))
                    obj.appendChild(elem)
                # bndbox
                bndbox = xmlBuilder.createElement("bndbox")
                x_center_f = float(x_center) * Pwidth + 1
                y_center_f = float(y_center) * Pheight + 1
                half_w = float(w_norm) * 0.5 * Pwidth
                half_h = float(h_norm) * 0.5 * Pheight
                coords = {
                    "xmin": int(max(0, x_center_f - half_w)),
                    "ymin": int(max(0, y_center_f - half_h)),
                    "xmax": int(min(Pwidth, x_center_f + half_w)),
                    "ymax": int(min(Pheight, y_center_f + half_h)),
                }
                for tag, val in coords.items():
                    elem = xmlBuilder.createElement(tag)
                    elem.appendChild(xmlBuilder.createTextNode(str(val)))
                    bndbox.appendChild(elem)
                obj.appendChild(bndbox)
                annotation.appendChild(obj)

            with open(xml_file, 'w', encoding='utf-8') as f:
                xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t')

            message = f"Converted successfully: {name} -> {xml_name}"
            # logs.append(f"Converted successfully: {name} -> {xml_name}")
        except Exception as e:
            # logs.append(f"Converted failed: {name} -> {xml_name}")
            message = f"Converted failed: {name} Error: {e}"

        update_callback(message)
        QCoreApplication.processEvents()
