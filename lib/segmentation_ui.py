import sys
from PySide6.QtCore import QFile, Qt, QCoreApplication
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox
)
from PySide6.QtGui import QIcon
from lib.function.seg_coco_to_labelme_json import coco_to_labelme_json
from lib.function.seg_coco_to_yolo_txt import coco_to_yolo_txt
from lib.function.seg_labelme_json_to_coco import labelme_json_to_coco


class Segmentation_Ui(QMainWindow):
    def __init__(self):
        super(Segmentation_Ui, self).__init__()

        qfile = QFile("ui/segmentation_ui.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()
        self.ui = QUiLoader().load(qfile)
        if not self.ui:
            print(QUiLoader().errorString())
            sys.exit(-1)

        self.input_coco_json_dir1 = ''
        self.output_labelme_json_dir = ''
        self.input_coco_json_dir2 = ''
        self.output_yolo_txt_dir= ''
        self.input_images_dir = ''
        self.input_labelme_json = ''
        self.output_coco_json = ''

        self.ui.setWindowIcon(QIcon("icon/DL-logo.svg"))

        self.ui.pushButton_1.clicked.connect(self.choose_input_cocojson_dir1)
        self.ui.pushButton_2.clicked.connect(self.choose_output_json_dir)
        self.ui.pushButton_start_2.clicked.connect(self.start_coco_json_to_labelme_json)

        self.ui.pushButton_3.clicked.connect(self.choose_input_cocojson_dir2)
        self.ui.pushButton_4.clicked.connect(self.choose_output_txt_dir)
        self.ui.pushButton_start_4.clicked.connect(self.start_coco_json_to_yolo_txt)

        self.ui.pushButton_5.clicked.connect(self.choose_input_image_dir)
        self.ui.pushButton_6.clicked.connect(self.choose_input_labelme_json)
        self.ui.pushButton_7.clicked.connect(self.choose_output_coco_json_dir)
        self.ui.pushButton_start_1.clicked.connect(self.start_labelme_json_to_coco_json)

        self.ui.pushButton_quit.clicked.connect(self.close_ui)

    def update_result(self, message):
        self.ui.textEdit_2.append(message)

    def choose_input_cocojson_dir1(self):
        dlg = QFileDialog(self, "选择输入 json 文件")  # 标题
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)  # 只能选择目录，不允许选单个文件
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, False)  # 是否只显示目录
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)  # 使用QT对话框
        dlg.setNameFilter("JSON File (*.json)")  # 设置过滤器
        dlg.setViewMode(QFileDialog.ViewMode.Detail)  # 显示细节，List不显示细节
        dlg.setLabelText(QFileDialog.DialogLabel.Accept, "确认")
        dlg.setLabelText(QFileDialog.DialogLabel.Reject, "取消")
        dlg.setLabelText(QFileDialog.DialogLabel.LookIn, "查看：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileName, "名称：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileType, "类型：")

        if dlg.exec():
            folder1 = dlg.selectedFiles()[0]
            self.input_coco_json_dir1 = folder1
            self.ui.lineEdit_1.setText(folder1)


    def choose_output_json_dir(self):
        dlg = QFileDialog(self, "选择输出 json 文件夹, 尽量与图片一起存放, 便于 labelme 查看")
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, False)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setNameFilter("Image Files (*.png *.jpg *.jpeg *.bmp)")
        dlg.setViewMode(QFileDialog.ViewMode.Detail)
        dlg.setLabelText(QFileDialog.DialogLabel.Accept, "确认")
        dlg.setLabelText(QFileDialog.DialogLabel.Reject, "取消")
        dlg.setLabelText(QFileDialog.DialogLabel.LookIn, "查看：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileName, "名称：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileType, "类型：")

        if dlg.exec():
            self.output_labelme_json_dir = dlg.selectedFiles()[0]
            self.ui.lineEdit_2.setText(self.output_labelme_json_dir)


    def choose_input_cocojson_dir2(self):
        dlg = QFileDialog(self, "选择输入 json 文件")  # 标题
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)  # 只能选择目录，不允许选单个文件
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, False)  # 是否只显示目录
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)  # 使用QT对话框
        dlg.setNameFilter("JSON File (*.json)")  # 设置过滤器
        dlg.setViewMode(QFileDialog.ViewMode.Detail)  # 显示细节，List不显示细节
        dlg.setLabelText(QFileDialog.DialogLabel.Accept, "确认")
        dlg.setLabelText(QFileDialog.DialogLabel.Reject, "取消")
        dlg.setLabelText(QFileDialog.DialogLabel.LookIn, "查看：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileName, "名称：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileType, "类型：")

        if dlg.exec():
            folder1 = dlg.selectedFiles()[0]
            self.input_coco_json_dir2 = folder1
            self.ui.lineEdit_9.setText(folder1)

    def choose_output_txt_dir(self):
        dlg = QFileDialog(self, "选择输出 TXT 文件目录")
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
            self.output_yolo_txt_dir = folder2
            self.ui.lineEdit_8.setText(folder2)

    def choose_input_image_dir(self):
        dlg = QFileDialog(self, "选择输入图片文件夹")
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
            self.input_images_dir = dlg.selectedFiles()[0]
            self.ui.lineEdit_3.setText(self.input_images_dir)


    def choose_input_labelme_json(self):
        dlg = QFileDialog(self, "选择输入json文件夹")
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, False)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setNameFilter("JSON (*.json)")
        dlg.setViewMode(QFileDialog.ViewMode.Detail)

        dlg.setLabelText(QFileDialog.DialogLabel.Accept, "确认")
        dlg.setLabelText(QFileDialog.DialogLabel.Reject, "取消")
        dlg.setLabelText(QFileDialog.DialogLabel.LookIn, "查看：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileName, "名称：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileType, "类型：")

        if dlg.exec():
            self.input_labelme_json = dlg.selectedFiles()[0]
            self.ui.lineEdit_4.setText(self.input_labelme_json)


    def choose_output_coco_json_dir(self):

        if not self.input_images_dir or not self.input_labelme_json:
            QMessageBox.warning(self, "错误", "请先选择图像路径和json路径")
            return

        dlg = QFileDialog(self, "选择保存 json 位置")
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        dlg.setNameFilter("JSON Files (*.json)")
        dlg.selectFile("annotations.json")
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setViewMode(QFileDialog.ViewMode.Detail)
        dlg.setLabelText(QFileDialog.DialogLabel.Accept, "确认")
        dlg.setLabelText(QFileDialog.DialogLabel.Reject, "取消")
        dlg.setLabelText(QFileDialog.DialogLabel.LookIn, "查看：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileName, "名称：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileType, "类型：")

        if dlg.exec():
            self.output_coco_json = dlg.selectedFiles()[0]
            self.ui.lineEdit_5.setText(self.output_coco_json)


    def start_coco_json_to_labelme_json(self):

        if not self.input_coco_json_dir1 or not self.output_labelme_json_dir:
            QMessageBox.warning(self, "错误", "请选择输入和输出目录")
            return

        self.ui.textEdit_2.clear()

        coco_to_labelme_json(self.input_coco_json_dir1, self.output_labelme_json_dir, self.update_result)

        QMessageBox.information(self, "完成", "所有文件转换完成")

    def start_coco_json_to_yolo_txt(self):

         if not self.input_coco_json_dir2 or not self.output_yolo_txt_dir:
             QMessageBox.warning(self, "错误", "请选择输入和输出目录")
             return

         self.ui.textEdit_2.clear()

         coco_to_yolo_txt(self.input_coco_json_dir2, self.output_yolo_txt_dir, self.update_result)

         QMessageBox.information(self, "完成", "所有文件转换完成")

    def start_labelme_json_to_coco_json(self):

        if not self.input_images_dir or not self.input_labelme_json or not self.output_coco_json:
            QMessageBox.warning(self, "错误", "请选择输入和输出目录")
            return

        self.ui.textEdit_2.clear()

        labelme_json_to_coco(self.input_labelme_json, self.input_images_dir,
                             self.output_coco_json, self.update_result)

        QMessageBox.information(self, "完成", "所有文件转换完成")

    def close_ui(self):
        self.ui.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Segmentation_Ui()
    window.ui.show()
    sys.exit(app.exec())

