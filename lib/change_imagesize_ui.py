import os
import sys
from PIL import Image
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QMessageBox, QMainWindow
)
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QIcon

def resize_images(input_folder, output_folder, scale_factor, output_format, update_callback):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + output_format
            output_path = os.path.join(output_folder, output_filename)

            try:
                with Image.open(input_path) as img:
                    original_size = img.size
                    new_size = (int(original_size[0] * scale_factor),
                                int(original_size[1] * scale_factor))
                    img_resized = img.resize(new_size)
                    img_resized = img_resized.convert('RGB')

                    if output_format.lower() in [".jpg", ".jpeg"]:
                        img_resized.save(output_path, format="JPEG")

                    else:
                        img_resized.save(output_path, format=output_format.strip('.').upper())
                    message = f"{filename}: {original_size} -> {new_size}, saved as {output_format}"
            except Exception as e:
                message = f"Processing {filename} Error: {e}"

            update_callback(message)
            QCoreApplication.processEvents()


class Change_imagesize_ui(QMainWindow):
    def __init__(self):
        super(Change_imagesize_ui, self).__init__()

        qfile = QFile("ui/change_imagesize_ui.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()

        self.ui = QUiLoader().load(qfile)

        if not self.ui:
            print(QUiLoader().errorString())
            sys.exit(-1)

        self.input_folder = ""
        self.output_folder = ""
        self.single_image_path = ""
        self.single_output_path = ""

        self.ui.setWindowIcon(QIcon("icon/DL-logo.svg"))

        self.ui.pushButton_1.clicked.connect(self.choose_input_folder)
        self.ui.pushButton_2.clicked.connect(self.choose_output_folder)
        self.ui.pushButton_3.clicked.connect(self.start_process)

        self.ui.pushButton_4.clicked.connect(self.choose_single_image)
        self.ui.pushButton_5.clicked.connect(self.choose_single_save_location)
        self.ui.pushButton_6.clicked.connect(self.start_single_process)
        self.ui.pushButton_7.clicked.connect(self.close_ui)

    ## 简单的对话框
    # def choose_input_folder(self):
    #     folder1 = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
    #     if folder1:
    #         self.input_folder = folder1
    #         self.ui.lineEdit.setText(folder1)
    #
    # def choose_output_folder(self):
    #     folder2 = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
    #     if folder2:
    #         self.output_folder = folder2
    #         self.ui.lineEdit_2.setText(folder2)

    ## 使用qt自带对话框
    def choose_input_folder(self):
        dlg = QFileDialog(self, "选择输入文件夹")
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, False)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setNameFilter("图片文件 (*.png *.jpg *.jpeg *.bmp)")
        dlg.setViewMode(QFileDialog.ViewMode.Detail)
        dlg.setLabelText(QFileDialog.DialogLabel.Accept, "确认")
        dlg.setLabelText(QFileDialog.DialogLabel.Reject, "取消")
        dlg.setLabelText(QFileDialog.DialogLabel.LookIn, "查看：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileName, "名称：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileType, "类型：")

        if dlg.exec():
            folder1 = dlg.selectedFiles()[0]
            self.input_folder = folder1
            self.ui.lineEdit.setText(folder1)

    def choose_output_folder(self):
        dlg = QFileDialog(self, "选择输出文件夹")
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
            folder2 = dlg.selectedFiles()[0]
            self.output_folder = folder2
            self.ui.lineEdit_2.setText(folder2)

    # def choose_single_image(self):
    #     filename, _ = QFileDialog.getOpenFileName(
    #         self, "选择图像", "", "Images (*.png *.jpg *.jpeg *.bmp)")
    #     if filename:
    #         self.single_image_path = filename
    #
    #         self.ui.lineEdit_3.setText(filename)

    def choose_single_image(self):
        dlg = QFileDialog(self, "选择图像")
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, False)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        dlg.setViewMode(QFileDialog.ViewMode.Detail)
        dlg.setLabelText(QFileDialog.DialogLabel.Accept, "确认")
        dlg.setLabelText(QFileDialog.DialogLabel.Reject, "取消")
        dlg.setLabelText(QFileDialog.DialogLabel.LookIn, "查看：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileName, "名称：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileType, "类型：")

        if dlg.exec():
            self.single_image_path = dlg.selectedFiles()[0]
            self.ui.lineEdit_3.setText(self.single_image_path)


    # def choose_single_save_location(self):
    #     # 选择单张图像保存位置
    #     # 注意：保存文件名可由用户自定义，也可根据原文件名自动生成
    #     save_filename, _ = QFileDialog.getSaveFileName(
    #         self, "选择保存位置", "", "Images (*{0})".format(self.ui.comboBox.currentText()))
    #     if save_filename:
    #         self.single_output_path = save_filename
    #         self.ui.lineEdit_4.setText(save_filename)


    def choose_single_save_location(self):
        if not self.single_image_path:
            QMessageBox.warning(self, "错误", "请先选择图像文件")
            return
        # 默认文件名为原文件名
        default_name = os.path.basename(self.single_image_path)
        dlg = QFileDialog(self, "选择保存位置")
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
            self.single_output_path = dlg.selectedFiles()[0]
            self.ui.lineEdit_4.setText(self.single_output_path)


    def update_result(self, message):
        self.ui.textEdit_2.append(message)

    def start_process(self):
        if not self.input_folder or not self.output_folder:
            QMessageBox.warning(self, "错误", "请选择输入和输出文件夹")
            return

        try:
            scale_factor = float(self.ui.doubleSpinBox.value())
        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的缩放比例")
            return

        output_format = self.ui.comboBox_2.currentText()
        if output_format not in [".jpg", ".jpeg", ".png", ".bmp"]:
            QMessageBox.warning(self, "错误", "请选择有效的图片格式")
            return

        self.ui.textEdit_2.clear()

        resize_images(input_folder=self.input_folder,
                      output_folder=self.output_folder,
                      scale_factor=scale_factor,
                      output_format=output_format,
                      update_callback=self.update_result)
        QMessageBox.information(self, "处理结果", "图片处理完成！")

    def start_single_process(self):

        if not self.single_image_path or not self.single_output_path:
            QMessageBox.warning(self, "错误", "请选择图像文件和保存位置")
            return

        try:
            scale_factor = float(self.ui.doubleSpinBox_2.value())
        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的缩放比例")
            return

        # Pillow 要求 JPEG 不能写成 JPG
        fmt = os.path.splitext(self.single_output_path)[1].strip('.').upper()
        if fmt == "JPG":
            fmt = "JPEG"

        self.ui.textEdit_2.clear()

        try:
            with Image.open(self.single_image_path) as img:
                original_size = img.size
                new_size = (int(original_size[0] * scale_factor),
                            int(original_size[1] * scale_factor))
                img_resized = img.resize(new_size).convert('RGB')
                img_resized.save(self.single_output_path, format=fmt)
                message = f"单张图片: {os.path.basename(self.single_image_path)} {original_size} -> {new_size}, saved to {self.single_output_path}"
        except Exception as e:
            message = f"Processing {os.path.basename(self.single_image_path)} Error: {e}"

        self.ui.textEdit_2.append(message)
        QMessageBox.information(self, "处理结果", "单张图片处理完成！")

    def close_ui(self):
        self.ui.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Change_imagesize_ui()
    window.ui.show()
    sys.exit(app.exec())
