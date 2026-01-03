import sys
import os
import cv2
import tempfile
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QMessageBox, QMainWindow
)
from PySide6.QtCore import QFile, Qt, QSize
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPixmap, QImage, QIcon

from lib.function.view_imgae_window import ViewImageWindow

class Style_transfer_ui(QMainWindow):
    def __init__(self):
        super(Style_transfer_ui, self).__init__()

        qfile = QFile("ui/image_style_transfer.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()

        self.ui = QUiLoader().load(qfile)

        if not self.ui:
            print(QUiLoader().errorString())
            sys.exit(-1)

        self.input_content_dir = ''
        self.input_style_dir = ''
        self.output_image_dir = ''
        self.content_paths = []
        self.style_paths = []
        self.current_content_index = -1
        self.current_style_index = -1

        self.output_img = None

        self.output_path = ''
        self.output_pixmap = None
        self.default_name = ''

        self.open_windows = []

        self.ui.setWindowIcon(QIcon("icon/DL-logo.svg"))

        self.ui.pushButton_1.clicked.connect(self.choose_input_content_folder)
        self.ui.pushButton_2.clicked.connect(self.choose_input_style_folder)
        self.ui.pushButton_3.clicked.connect(self.choose_output_image_folder)


        self.ui.pushButton_style_transfer.clicked.connect(self.start_batch_image_style_transfer)
        self.ui.pushButton_style_transfer_1.clicked.connect(self.start_single_imgae_style_transfer)

        self.ui.pushButton_save_1.clicked.connect(self.on_save_image)

        self.ui.pushButton_quit.clicked.connect(self.close_ui)

        self.ui.pushButton_left.setIcon(QIcon("icon/left.svg"))
        self.ui.pushButton_left.setIconSize(QSize(40, 20))
        self.ui.pushButton_left.clicked.connect(self.show_previous_content)
        self.ui.pushButton_right.setIcon(QIcon("icon/right.svg"))
        self.ui.pushButton_right.setIconSize(QSize(40, 20))
        self.ui.pushButton_right.clicked.connect(self.show_next_content)

        self.ui.pushButton_left_1.setIcon(QIcon("icon/left.svg"))
        self.ui.pushButton_left_1.setIconSize(QSize(40, 20))
        self.ui.pushButton_left_1.clicked.connect(self.show_previous_style)
        self.ui.pushButton_right_1.setIcon(QIcon("icon/right.svg"))
        self.ui.pushButton_right_1.setIconSize(QSize(40, 20))
        self.ui.pushButton_right_1.clicked.connect(self.show_next_style)

        self.ui.pushButton_fs_1.clicked.connect(lambda: self.show_full_image("content"))
        self.ui.pushButton_fs_2.clicked.connect(lambda: self.show_full_image("style"))
        self.ui.pushButton_fs_3.clicked.connect(lambda: self.show_full_image("output"))


    ######################    增强预览代码，单独增强  #################
    def show_full_image(self, img_type: str):

        if img_type == "content":
            if self.current_content_index < 0 or not self.content_paths:
                QMessageBox.warning(self, "错误", "请先选择Content图像")
                return
            img_path = self.content_paths[self.current_content_index]

        elif img_type == "style":
            if self.current_style_index < 0 or not self.style_paths:
                QMessageBox.warning(self, "错误", "请先选择Style图像")
                return
            img_path = self.style_paths[self.current_style_index]

        elif img_type == "output":
            if self.output_img is None:
                QMessageBox.warning(self, "错误", "当前无可预览的输出图像")
                return
        else:
            QMessageBox.warning(self, "错误", f"未知的预览类型：{img_type}")
            return

        if img_type in ("content", "style"):
            img = cv2.imread(img_path)
            if img is None:
                QMessageBox.warning(self, "错误", f"无法加载图像: {img_path}")
                return
        else:
            img = self.output_img

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img)

        title_map = {
            "content": "content图像",
            "style": "style图像",
            "output": "style-transfer图像",
        }

        self.window = ViewImageWindow(pix)
        self.window.setWindowTitle(title_map.get(img_type, "View image"))
        self.window.setWindowIcon(QIcon("icon/full_screen.svg"))

        self.open_windows.append(self.window)
        self.window.setWindowModality(Qt.ApplicationModal)
        self.window.show()

    def choose_input_content_folder(self):
        dlg = QFileDialog(self, "选择输入Content图像文件目录")  # 标题
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
            self.input_content_dir = dlg.selectedFiles()[0]
            self.ui.lineEdit_1.setText(self.input_content_dir)
            self.load_content_images()

    def choose_input_style_folder(self):
        dlg = QFileDialog(self, "选择输入Style图像文件目录")  # 标题
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
            self.input_style_dir = dlg.selectedFiles()[0]
            self.ui.lineEdit_2.setText(self.input_style_dir)
            self.load_style_images()

    def choose_output_image_folder(self):
        dlg = QFileDialog(self, "选择输出文件目录")
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

    def update_result(self, message):
        self.ui.textEdit_2.append(message)


    def load_content_images(self):
        supported_ext = ['.png', '.jpg', '.jpeg', '.bmp']
        all_files = os.listdir(self.input_content_dir)
        self.content_paths = [
            os.path.join(self.input_content_dir, f)
            for f in sorted(all_files)
            if os.path.splitext(f)[1].lower() in supported_ext
        ]
        if self.content_paths:
            self.current_content_index = 0
            self.show_current_content()
        else:
            self.current_content_index = -1
            self.ui.label_content.clear()
            QMessageBox.warning(self, "错误", "未在目录中找到图片文件")

    def show_current_content(self):
        if 0 <= self.current_content_index < len(self.content_paths):
            content_path = self.content_paths[self.current_content_index]
            pixmap = QPixmap(content_path)
            if pixmap.isNull():
                QMessageBox.warning(self, "错误", f"无法加载图像: {content_path}")
                self.ui.label_content.clear()
                return
            label_size = self.ui.label_content.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui.label_content.setPixmap(scaled_pixmap)
            filename = os.path.basename(content_path)
            self.update_result(f"image: {self.current_content_index + 1}/{len(self.content_paths)}: {filename}")
        else:
            self.ui.label_content.clear()


    def load_style_images(self):
        supported_ext = ['.png', '.jpg', '.jpeg', '.bmp']
        all_files = os.listdir(self.input_style_dir)
        self.style_paths = [
            os.path.join(self.input_style_dir, f)
            for f in sorted(all_files)
            if os.path.splitext(f)[1].lower() in supported_ext
        ]
        if self.style_paths:
            self.current_style_index = 0
            self.show_current_style()
        else:
            self.current_style_index = -1
            self.ui.label_style.clear()
            QMessageBox.warning(self, "错误", "未在目录中找到图片文件")

    def show_current_style(self):
        if 0 <= self.current_style_index < len(self.style_paths):
            style_path = self.style_paths[self.current_style_index]
            pixmap = QPixmap(style_path)
            if pixmap.isNull():
                QMessageBox.warning(self, "错误", f"无法加载图像: {style_path}")
                self.ui.label_style.clear()
                return
            label_size = self.ui.label_style.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui.label_style.setPixmap(scaled_pixmap)
            filename = os.path.basename(style_path)
            self.update_result(f"image: {self.current_style_index + 1}/{len(self.style_paths)}: {filename}")
        else:
            self.ui.label_style.clear()


    def show_previous_content(self):
        if not self.content_paths:
            QMessageBox.warning(self, "错误", "请选择输入Content图像目录")
            return
        self.current_content_index = (self.current_content_index - 1) % len(self.content_paths)
        self.ui.textEdit_2.clear()
        # self.window.close()
        self.show_current_content()


    def show_next_content(self):
        if not self.content_paths:
            QMessageBox.warning(self, "错误", "请选择输入Content图像目录")
            return
        self.current_content_index = (self.current_content_index + 1) % len(self.content_paths)
        self.ui.textEdit_2.clear()
        # self.window.close()
        self.show_current_content()


    def show_previous_style(self):
        if not self.style_paths:
            QMessageBox.warning(self, "错误", "请选择输入Style图像目录")
            return
        self.current_style_index = (self.current_style_index - 1) % len(self.style_paths)
        self.ui.textEdit_2.clear()
        # self.window.close()
        self.show_current_style()

    def show_next_style(self):
        if not self.style_paths:
            QMessageBox.warning(self, "错误", "请选择输入Style图像目录")
            return
        self.current_style_index = (self.current_style_index + 1) % len(self.style_paths)
        self.ui.textEdit_2.clear()
        # self.window.close()
        self.show_current_style()

    def start_batch_image_style_transfer(self):

        if not self.input_content_dir or not self.input_style_dir:
            QMessageBox.warning(self, "错误","请先选择Content和Style目录")
            return

        if not self.output_image_dir:
            QMessageBox.warning(self, "错误","请选择输出目录")
            return

        transfer_mode = self.ui.comboBox_1.currentText()
        self.ui.textEdit_2.clear()
        self.update_result(f"模型：{transfer_mode}\n")

        if transfer_mode == 'S2WAT':
            from .function.Image_Style_Transfer.S2WAT.image_style_transfer import s2wat_transfer_imgs

            s2wat_transfer_imgs(content_dir=self.input_content_dir,
                                style_dir=self.input_style_dir,
                                output_dir=self.output_image_dir,
                                update_callback=self.update_result)

        if transfer_mode == 'CAPVST_photorealistic':
            from .function.Image_Style_Transfer.CAPVSTNet.image_transfer_batch import cap_vstnet_images_transfer

            cap_vstnet_images_transfer(mode='photorealistic',
                                       content_dir=self.input_content_dir,
                                       style_dir=self.input_style_dir,
                                       out_dir=self.output_image_dir,
                                       update_callback=self.update_result)
            QMessageBox.information(self, "完成", "风格迁移完成")

        if transfer_mode == 'CAPVST_artistic':
            from .function.Image_Style_Transfer.CAPVSTNet.image_transfer_batch import cap_vstnet_images_transfer

            cap_vstnet_images_transfer(mode='artistic',
                                       content_dir=self.input_content_dir,
                                       style_dir=self.input_style_dir,
                                       out_dir=self.output_image_dir,
                                       update_callback=self.update_result)
            QMessageBox.information(self, "完成", "风格迁移完成")

        if transfer_mode == 'ArtFlow_adain':
            from .function.Image_Style_Transfer.ArtFlow.ArtFlow_batch_style_transfer import ArtFlow_batch

            ArtFlow_batch(mode='adain',
                          content_dir=self.input_content_dir,
                          style_dir=self.input_style_dir,
                          output_dir=self.output_image_dir,
                          update_callback=self.update_result)
            QMessageBox.information(self, "完成", "风格迁移完成")

        if transfer_mode == 'ArtFlow_wct':
            from .function.Image_Style_Transfer.ArtFlow.ArtFlow_batch_style_transfer import ArtFlow_batch

            ArtFlow_batch(mode='wct',
                          content_dir=self.input_content_dir,
                          style_dir=self.input_style_dir,
                          output_dir=self.output_image_dir,
                          update_callback=self.update_result)
            QMessageBox.information(self, "完成", "风格迁移完成")

        if transfer_mode == 'ArtFlow_portrait':
            from .function.Image_Style_Transfer.ArtFlow.ArtFlow_batch_style_transfer import ArtFlow_batch

            ArtFlow_batch(mode='portrait',
                          content_dir=self.input_content_dir,
                          style_dir=self.input_style_dir,
                          output_dir=self.output_image_dir,
                          update_callback=self.update_result)
            QMessageBox.information(self, "完成", "风格迁移完成")

    def show_output_image(self, image_path):

        img = cv2.imread(image_path)
        if img is None:
            QMessageBox.warning(self, "错误", f"无法加载生成结果: {image_path}")
            return
        self.output_img = img  # 给self.output_img赋值

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "错误", f"无法加载生成结果: {image_path}")
            return

        self.ui.label_output.setPixmap(pixmap.scaled(
            self.ui.label_output.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.output_pixmap = pixmap
        self.output_path = image_path

    def on_save_image(self):
        if not self.output_pixmap or not self.output_path:
            QMessageBox.warning(self, "错误", "当前无可保存的图像")
            return
        default_name = self.default_name if self.default_name else os.path.basename(self.output_path)
        self.save_image(self.output_pixmap, default_name)

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
            save_path = dlg.selectedFiles()[0]
            success = img.save(save_path)
            if not success:
                QMessageBox.warning(self, "错误", f"图像保存失败: {save_path}")
            else:
                self.update_result(f"Saved to: {save_path}")
                QMessageBox.information(self, "提示", f"已保存 {save_path}")


    def start_single_imgae_style_transfer(self):

        if not self.input_content_dir or not self.input_style_dir:
            QMessageBox.warning(self, "错误", "请先选择Content和Style目录")
            return

        transfer_mode = self.ui.comboBox_1.currentText()
        self.ui.textEdit_2.clear()
        self.update_result(f"模型：{transfer_mode}\n")

        content_path = self.content_paths[self.current_content_index]
        style_path = self.style_paths[self.current_style_index]
        base_content = os.path.splitext(os.path.basename(content_path))[0]
        base_style = os.path.splitext(os.path.basename(style_path))[0]
        default_name = f"{base_content}_{base_style}.jpg"
        self.default_name = default_name

        if transfer_mode == 'S2WAT':
            from .function.Image_Style_Transfer.S2WAT.single_image_style_transfer import s2wat_transfer_img

            tmp = tempfile.NamedTemporaryFile(delete=False,
                                              suffix=os.path.splitext(default_name)[1])
            tmp.close()
            s2wat_transfer_img(content_path=content_path,
                               style_path=style_path,
                               output_path=tmp.name,
                               update_callback=self.update_result)

            self.show_output_image(tmp.name)

        if transfer_mode == 'CAPVST_photorealistic':
            from .function.Image_Style_Transfer.CAPVSTNet.image_transfer_single import cap_vstnet_image_transfer

            tmp = tempfile.NamedTemporaryFile(delete=False,
                                              suffix=os.path.splitext(default_name)[1])
            tmp.close()
            cap_vstnet_image_transfer(mode='photorealistic',
                                      content_path=content_path,
                                      style_path=style_path,
                                      out_path=tmp.name,
                                      update_callback=self.update_result)
            self.show_output_image(tmp.name)

        if transfer_mode == 'CAPVST_artistic':
            from .function.Image_Style_Transfer.CAPVSTNet.image_transfer_single import cap_vstnet_image_transfer

            tmp = tempfile.NamedTemporaryFile(delete=False,
                                              suffix=os.path.splitext(default_name)[1])
            tmp.close()
            cap_vstnet_image_transfer(mode='artistic',
                                      content_path=content_path,
                                      style_path=style_path,
                                      out_path=tmp.name,
                                      update_callback=self.update_result)
            self.show_output_image(tmp.name)

        if transfer_mode == 'ArtFlow_adain':
            from .function.Image_Style_Transfer.ArtFlow.ArtFlow_single_style_transfer import ArtFlow_single

            tmp = tempfile.NamedTemporaryFile(delete=False,
                                              suffix=os.path.splitext(default_name)[1])
            tmp.close()
            ArtFlow_single(mode='adain',
                           content_path=content_path,
                           style_path=style_path,
                           output_path=tmp.name,
                           update_callback=self.update_result)
            self.show_output_image(tmp.name)

        if transfer_mode == 'ArtFlow_wct':
            from .function.Image_Style_Transfer.ArtFlow.ArtFlow_single_style_transfer import ArtFlow_single
            tmp = tempfile.NamedTemporaryFile(delete=False,
                                              suffix=os.path.splitext(default_name)[1])
            tmp.close()
            ArtFlow_single(mode='wct',
                           content_path=content_path,
                           style_path=style_path,
                           output_path=tmp.name,
                           update_callback=self.update_result)
            self.show_output_image(tmp.name)

        if transfer_mode == 'ArtFlow_portrait':
            from .function.Image_Style_Transfer.ArtFlow.ArtFlow_single_style_transfer import ArtFlow_single

            tmp = tempfile.NamedTemporaryFile(delete=False,
                                              suffix=os.path.splitext(default_name)[1])
            tmp.close()
            ArtFlow_single(mode='portrait',
                           content_path=content_path,
                           style_path=style_path,
                           output_path=tmp.name,
                           update_callback=self.update_result)
            self.show_output_image(tmp.name)

    def close_ui(self):
        self.ui.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Style_transfer_ui()
    window.ui.show()
    sys.exit(app.exec())


