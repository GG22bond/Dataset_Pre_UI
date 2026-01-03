import cv2
import sys
import os
import numpy as np
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt, QSize
from PySide6.QtGui import QPixmap, QImage, QIcon
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QMessageBox, QMainWindow
)

from lib.function.amp import DataAugmentForObjectDetection, ToolHelper
from lib.function.amp_image_single import Single_DataAugmentForObjectDetection, Single_ToolHelper
from lib.function.amp_config_ui import Amp_config_ui
from lib.function.view_imgae_window import ViewImageWindow

class Amplification_ui(QMainWindow):
    def __init__(self):
        super(Amplification_ui, self).__init__()

        # 加载主界面 UI
        qfile = QFile("ui/amp_ui.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()
        self.ui = QUiLoader().load(qfile)
        if not self.ui:
            print(QUiLoader().errorString())
            sys.exit(-1)

        # 默认参数
        self.params = [0.01, 0.6, 30, 1, 0.3, 5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.config_dialog = None
        self.input_image_dir = ''
        self.input_xml_dir = ''
        self.output_image_dir = ''
        self.output_xml_dir = ''
        self.image_paths = []
        self.xml_paths = []
        self.current_image_index = -1
        self.current_xml_index = -1

        self.noise_img = None
        self.bright_img = None
        self.cutout_img = None
        self.rotate_img = None
        self.shift_img = None
        self.flip_img = None

        self.open_windows = []

        self.ui.setWindowIcon(QIcon("icon/DL-logo.svg"))

        # 绑定按钮
        # 文件路径
        self.ui.pushButton_1.clicked.connect(self.choose_image_input_dir)
        self.ui.pushButton_2.clicked.connect(self.choose_xml_input_dir)
        self.ui.pushButton_3.clicked.connect(self.choose_image_output_dir)
        self.ui.pushButton_4.clicked.connect(self.choose_xml_output_dir)

        # 参数设置
        self.ui.pushButton_5.clicked.connect(self.open_config_dialog)

        # 增强预览
        self.ui.pushButton_left.setIcon(QIcon("icon/left.svg"))
        self.ui.pushButton_left.setIconSize(QSize(40, 20))
        self.ui.pushButton_left.clicked.connect(self.show_previous_image)

        self.ui.pushButton_right.setIcon(QIcon("icon/right.svg"))
        self.ui.pushButton_right.setIconSize(QSize(40, 20))
        self.ui.pushButton_right.clicked.connect(self.show_next_image)
        self.ui.pushButton_9.clicked.connect(self.start_single_amp)
        # 保存预览
        self.ui.pushButton_save_1.clicked.connect(self.save_noise)
        self.ui.pushButton_save_2.clicked.connect(self.save_bright)
        self.ui.pushButton_save_3.clicked.connect(self.save_cutout)
        self.ui.pushButton_save_4.clicked.connect(self.save_rotate)
        self.ui.pushButton_save_5.clicked.connect(self.save_shift)
        self.ui.pushButton_save_6.clicked.connect(self.save_flip)
        # 选择放大预览
        self.ui.pushButton_fs_1.clicked.connect(lambda: self.show_full_image("noise"))
        self.ui.pushButton_fs_2.clicked.connect(lambda: self.show_full_image("bright"))
        self.ui.pushButton_fs_3.clicked.connect(lambda: self.show_full_image("cutout"))
        self.ui.pushButton_fs_4.clicked.connect(lambda: self.show_full_image("rotate"))
        self.ui.pushButton_fs_5.clicked.connect(lambda: self.show_full_image("shift"))
        self.ui.pushButton_fs_6.clicked.connect(lambda: self.show_full_image("flip"))
        self.ui.pushButton_fs_7.clicked.connect(lambda: self.show_full_image("original"))

        # 开始数据增强
        self.ui.pushButton_start.clicked.connect(self.start_amp)
        # 关闭窗口
        self.ui.pushButton_quit.clicked.connect(self.close_ui)
        self.ui.label_image.setScaledContents(False)

    def close_all_windows(self):
        for window in self.open_windows:
            try:
                window.close()
            except:
                pass
        self.open_windows.clear()

    ######################    增强预览代码，单独增强  #################
    def show_full_image(self, img_type: str):
        if self.current_image_index < 0 or not self.image_paths:
            QMessageBox.warning(self, "错误", "请先选择图像目录")
            return

        if img_type == "original":
            img_path = self.image_paths[self.current_image_index]
            img = cv2.imread(img_path)
            if img is None:
                QMessageBox.warning(self, "错误", f"无法加载图像: {img_path}")
                return
        else:
            img = {
                "noise": self.noise_img,
                "bright": self.bright_img,
                "cutout": self.cutout_img,
                "rotate": self.rotate_img,
                "shift": self.shift_img,
                "flip": self.flip_img
            }.get(img_type)
            if img is None:
                QMessageBox.warning(self, "错误", f"未选择 {img_type} 增强方式")
                return

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img)

        title_map = {
            "noise": "noise处理图像",
            "bright": "bright处理图像",
            "cutout": "cutout处理图像",
            "rotate": "rotate处理图像",
            "shift": "shift处理图像",
            "flip": "flip处理图像",
            "original": "original image"
        }

        self.window = ViewImageWindow(pix)
        self.window.setWindowTitle(title_map.get(img_type, "View image"))
        self.window.setWindowIcon(QIcon("icon/full_screen.svg"))

        self.open_windows.append(self.window)
        self.window.setWindowModality(Qt.ApplicationModal)
        self.window.show()

    def choose_image_input_dir(self):
        dlg = QFileDialog(self, "选择输入图像文件目录")  # 标题
        dlg.setWindowIcon(QIcon("icon/image-folder.svg"))
        dlg.setFileMode(QFileDialog.FileMode.Directory)  # 只能选择目录，不允许选单个文件
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, False)  # 是否只显示目录
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)  # 使用QT对话框
        dlg.setNameFilter("图片文件 (*.png *.jpg *.jpeg *.bmp)")  # 设置过滤器
        dlg.setViewMode(QFileDialog.ViewMode.Detail)  # 显示细节，List不显示细节
        dlg.setLabelText(QFileDialog.DialogLabel.Accept, "确认")
        dlg.setLabelText(QFileDialog.DialogLabel.Reject, "取消")
        dlg.setLabelText(QFileDialog.DialogLabel.LookIn, "查看：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileName, "名称：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileType, "类型：")

        if dlg.exec():
            self.input_image_dir = dlg.selectedFiles()[0]
            self.ui.lineEdit.setText(self.input_image_dir)
            self.load_images()

    def choose_xml_input_dir(self):
        dlg = QFileDialog(self, "选择输入 XML 文件目录")  # 标题
        dlg.setWindowIcon(QIcon("icon/xml-folder.svg"))
        dlg.setFileMode(QFileDialog.FileMode.Directory)  # 只能选择目录，不允许选单个文件
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, False)  # 是否只显示目录
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)  # 使用QT对话框
        dlg.setNameFilter("XML文件 (*.xml)")  # 设置过滤器
        dlg.setViewMode(QFileDialog.ViewMode.Detail)  # 显示细节，List不显示细节
        dlg.setLabelText(QFileDialog.DialogLabel.Accept, "确认")
        dlg.setLabelText(QFileDialog.DialogLabel.Reject, "取消")
        dlg.setLabelText(QFileDialog.DialogLabel.LookIn, "查看：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileName, "名称：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileType, "类型：")

        if dlg.exec():
            self.input_xml_dir = dlg.selectedFiles()[0]
            self.ui.lineEdit_2.setText(self.input_xml_dir)
            self.load_xmls()
            self.show_current_xml()

    def choose_image_output_dir(self):
        dlg = QFileDialog(self, "选择输出图像文件目录")
        dlg.setWindowIcon(QIcon("icon/image-folder.svg"))
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setOption(QFileDialog.Option.HideNameFilterDetails, True)
        dlg.setViewMode(QFileDialog.ViewMode.Detail)
        dlg.setLabelText(QFileDialog.DialogLabel.Accept, "确认")
        dlg.setLabelText(QFileDialog.DialogLabel.Reject, "取消")
        dlg.setLabelText(QFileDialog.DialogLabel.LookIn, "查看：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileName, "名称：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileType, "类型：")

        if dlg.exec():
            self.output_image_dir = dlg.selectedFiles()[0]
            self.ui.lineEdit_3.setText(self.output_image_dir)

    def choose_xml_output_dir(self):
        dlg = QFileDialog(self, "选择输出 XML 文件目录")
        dlg.setWindowIcon(QIcon("icon/xml-folder.svg"))
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setOption(QFileDialog.Option.HideNameFilterDetails, True)
        dlg.setViewMode(QFileDialog.ViewMode.Detail)
        dlg.setLabelText(QFileDialog.DialogLabel.Accept, "确认")
        dlg.setLabelText(QFileDialog.DialogLabel.Reject, "取消")
        dlg.setLabelText(QFileDialog.DialogLabel.LookIn, "查看：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileName, "名称：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileType, "类型：")

        if dlg.exec():
            self.output_xml_dir = dlg.selectedFiles()[0]
            self.ui.lineEdit_4.setText(self.output_xml_dir)

    def open_config_dialog(self):
        if self.config_dialog is None:
            self.config_dialog = Amp_config_ui()
            # self.config_dialog.setWindowModality(Qt.ApplicationModal)
            self.config_dialog.result_ready.connect(self.update_params)
        self.config_dialog.ui.exec()

    def update_result(self, message):
        self.ui.textEdit_2.append(message)

    def load_images(self):
        supported_ext = ['.png', '.jpg', '.jpeg', '.bmp']
        all_files = os.listdir(self.input_image_dir)
        self.image_paths = [
            os.path.join(self.input_image_dir, f)
            for f in sorted(all_files)
            if os.path.splitext(f)[1].lower() in supported_ext
        ]
        if self.image_paths:
            self.current_image_index = 0
            self.show_current_image()
        else:
            self.current_image_index = -1
            self.ui.label_image.clear()
            QMessageBox.warning(self, "错误", "未在目录中找到图片文件")

    def load_xmls(self):
        supported_ext = ['.xml']
        all_files = os.listdir(self.input_xml_dir)
        self.xml_paths = [
            os.path.join(self.input_xml_dir, f)
            for f in sorted(all_files)
            if os.path.splitext(f)[1].lower() in supported_ext
        ]
        if self.xml_paths:
            self.current_xml_index = 0
        else:
            self.current_xml_index = -1
            QMessageBox.warning(self, "错误", "未在目录中找到xml文件")

    def show_current_image(self):
        if 0 <= self.current_image_index < len(self.image_paths):
            img_path = self.image_paths[self.current_image_index]
            pixmap = QPixmap(img_path)
            if pixmap.isNull():
                QMessageBox.warning(self, "错误", f"无法加载图像: {img_path}")
                self.ui.label_image.clear()
                return
            label_size = self.ui.label_image.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui.label_image.setPixmap(scaled_pixmap)
            filename = os.path.basename(img_path)
            self.update_result(f"image: {self.current_image_index + 1}/{len(self.image_paths)}: {filename}")
        else:
            self.ui.label_image.clear()

    def show_current_xml(self):
        if 0 <= self.current_xml_index < len(self.xml_paths):
            filename = os.path.basename(self.xml_paths[self.current_xml_index])
            self.update_result(f"xml: {self.current_xml_index + 1}/{len(self.xml_paths)}: {filename}")

    def show_previous_image(self):
        if not self.image_paths or not self.xml_paths:
            QMessageBox.warning(self, "错误", "请先选择图像目录和XML目录")
            return
        self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
        self.current_xml_index = (self.current_xml_index - 1) % len(self.xml_paths)
        self.ui.textEdit_2.clear()
        # self.window.close()
        self.show_current_image()
        self.show_current_xml()

    def show_next_image(self):
        if not self.image_paths or not self.xml_paths:
            QMessageBox.warning(self, "错误", "请先选择图像目录和XML目录")
            return
        self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
        self.current_xml_index = (self.current_xml_index + 1) % len(self.xml_paths)
        self.ui.textEdit_2.clear()
        # self.window.close()
        self.show_current_image()
        self.show_current_xml()

    def update_params(self, params_list):

        if isinstance(params_list, list) and len(params_list) == len(self.params):
            self.params = params_list
            self.update_result(f"params_configs：{self.params[:6]}")
        else:
            self.update_result("Error. Using default params.")


    def start_single_amp(self):
        if not self.image_paths or not self.xml_paths:
            QMessageBox.warning(self, "错误", "请先选择图像目录和XML目录")
            return

        flags = {
            'noise': self.ui.checkBox_noise.isChecked(),
            'bright': self.ui.checkBox_bright.isChecked(),
            'cutout': self.ui.checkBox_cutout.isChecked(),
            'rotate': self.ui.checkBox_rotate.isChecked(),
            'shift': self.ui.checkBox_shift.isChecked(),
            'flip': self.ui.checkBox_flip.isChecked()
        }
        if not any(flags.values()):
            QMessageBox.warning(self, "错误", "请选择增强方式")
            return

        img_path = self.image_paths[self.current_image_index]
        xml_path = self.xml_paths[self.current_xml_index]

        self.params_2 = self.params[:6]

        dataAug = Single_DataAugmentForObjectDetection(
            self.params_2,
            is_addNoise=flags['noise'],
            is_changeLight=flags['bright'],
            is_cutout=flags['cutout'],
            is_rotate_img_bbox=flags['rotate'],
            is_shift_pic_bboxes=flags['shift'],
            is_flip_pic_bboxes=flags['flip']
        )

        img = cv2.imread(img_path)

        if flags['noise']:
            self.noise_img = dataAug.add_noise(img).astype(img.dtype)
            self._show_on_label(self.noise_img, self.ui.label_noise)
        else:
            self.ui.label_noise.clear()

        if flags['bright']:
            self.bright_img = dataAug.change_light(img).astype(img.dtype)
            self._show_on_label(self.bright_img, self.ui.label_bright)
        else:
            self.ui.label_bright.clear()

        if flags['cutout']:
            xml_vals = Single_ToolHelper().parse_xml(xml_path)
            boxes = [v[:4] for v in xml_vals]
            self.cutout_img = dataAug.cutout(img, boxes).astype(img.dtype)
            self._show_on_label(self.cutout_img, self.ui.label_cutout)
        else:
            self.ui.label_cutout.clear()

        if flags['rotate']:
            xml_vals = Single_ToolHelper().parse_xml(xml_path)
            boxes = [v[:4] for v in xml_vals]
            rotated, _ = dataAug.rotate_img_bbox(img, boxes)
            self.rotate_img = rotated.astype(img.dtype)
            self._show_on_label(self.rotate_img, self.ui.label_rotate)
        else:
            self.ui.label_rotate.clear()

        if flags['shift']:
            xml_vals = Single_ToolHelper().parse_xml(xml_path)
            boxes = [v[:4] for v in xml_vals]
            shifted, _ = dataAug.shift_pic_bboxes(img, boxes)
            self.shift_img = shifted.astype(img.dtype)
            self._show_on_label(self.shift_img, self.ui.label_shift)
        else:
            self.ui.label_shift.clear()

        if flags['flip']:
            xml_vals = Single_ToolHelper().parse_xml(xml_path)
            boxes = [v[:4] for v in xml_vals]
            flipped, _ = dataAug.flip_pic_bboxes(img, boxes)
            self.flip_img = flipped.astype(img.dtype)
            self._show_on_label(self.flip_img, self.ui.label_flip)
        else:
            self.ui.label_flip.clear()

    def _show_on_label(self, img: np.ndarray, label):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pix)


    # def save_image(self, img, default_name):
    #     if img is None:
    #         QMessageBox.warning(self, "错误", "当前无可保存的图像")
    #         return
    #
    #     path, _ = QFileDialog.getSaveFileName(
    #         self, "保存图像", default_name,
    #         "JPEG Files (*.jpg *.jpeg);;PNG Files (*.png);;BMP Files (*.bmp);;All Files (*)")
    #
    #     if path:
    #         cv2.imwrite(path, img)
    #         QMessageBox.information(self, "提示", f"已保存 {path}")


    def save_image(self, img, default_name):
        if img is None:
            QMessageBox.warning(self, "错误", "当前无可保存的图像")
            return
        dlg = QFileDialog(self, "保存图像")
        dlg.setWindowIcon(QIcon("icon/save_image.svg"))
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        dlg.setNameFilter("JPEG Files (*.jpg *.jpeg);;PNG Files (*.png);;BMP Files (*.bmp);;All Files (*)")
        dlg.selectFile(default_name)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setViewMode(QFileDialog.ViewMode.Detail)
        dlg.setLabelText(QFileDialog.DialogLabel.Accept, "确认")
        dlg.setLabelText(QFileDialog.DialogLabel.Reject, "取消")
        dlg.setLabelText(QFileDialog.DialogLabel.LookIn, "查看：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileName, "名称：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileType, "类型：")

        if dlg.exec():
            path = dlg.selectedFiles()[0]
            cv2.imwrite(path, img)
            QMessageBox.information(self, "提示", f"已保存 {path}")


    def save_noise(self):
        self.save_image(self.noise_img, "noise.jpg")

    def save_bright(self):
        self.save_image(self.bright_img, "bright.jpg")

    def save_cutout(self):
        self.save_image(self.cutout_img, "cutout.jpg")

    def save_rotate(self):
        self.save_image(self.rotate_img, "rotate.jpg")

    def save_shift(self):
        self.save_image(self.shift_img, "shift.jpg")

    def save_flip(self):
        self.save_image(self.flip_img, "flip.jpg")


    #################################  批量数据增强   #####################################
    def start_amp(self):

        if (not self.input_image_dir or not self.input_xml_dir
                or not self.output_image_dir or not self.output_xml_dir):
            QMessageBox.warning(self, "错误", "请选择输入和输出目录")
            return

        try:
            amp_number = int(self.ui.spinBox.value())

        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的倍数")
            return

        addnoise = self.ui.checkBox_noise.isChecked()
        changelight = self.ui.checkBox_bright.isChecked()
        cutout = self.ui.checkBox_cutout.isChecked()
        rotate = self.ui.checkBox_rotate.isChecked()
        shift = self.ui.checkBox_shift.isChecked()
        flip = self.ui.checkBox_flip.isChecked()

        if not addnoise and not changelight and not cutout and not rotate and not shift and not flip:
            QMessageBox.warning(self, "错误", "请选择增强方式")
            return

        self.ui.textEdit_2.clear()

        self.update_result(f"amp number: {amp_number}")

        dataAug = DataAugmentForObjectDetection(params_list1=self.params,
                                                is_addNoise=addnoise,
                                                is_changeLight=changelight,
                                                is_cutout=cutout,
                                                is_rotate_img_bbox=rotate,
                                                is_shift_pic_bboxes=shift,
                                                is_flip_pic_bboxes=flip)

        toolhelper = ToolHelper()
        toolhelper.process_dataset(
            dataAug=dataAug,
            need_num=amp_number,
            source_img_path=self.input_image_dir,
            source_xml_path=self.input_xml_dir,
            save_img_path=self.output_image_dir,
            save_xml_path=self.output_xml_dir,
            update_callback=self.update_result
        )

        QMessageBox.information(self, "完成", "数据增强完成")

    def close_ui(self):
        self.ui.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Amplification_ui()
    window.ui.show()
    sys.exit(app.exec())
