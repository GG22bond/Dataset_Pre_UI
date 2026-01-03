import sys
import ast
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt, QCoreApplication
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox
)
from PySide6.QtGui import QIcon
from lib.function.classes_manage import Classes_ui
from lib.function.coco_to_yolo import convert_coco_to_yolo
from lib.function.yolo_to_coco import convert_yolo_to_coco


class COCO_and_YOLO(QMainWindow):
    def __init__(self):
        super(COCO_and_YOLO, self).__init__()

        qfile = QFile("ui/COCO_and_YOLO.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()
        self.ui = QUiLoader().load(qfile)
        if not self.ui:
            print(QUiLoader().errorString())
            sys.exit(-1)

        self.classes_manager = None

        # json to txt
        self.input_json_dir = ''
        self.output_txt_dir = ''

        # txt to json
        self.input_image_dir = ''
        self.input_txt_dir = ''
        self.output_json_dir = ''

        self.ui.setWindowIcon(QIcon("icon/DL-logo.svg"))

        self.ui.pushButton_class.clicked.connect(self.open_classes_dialog)
        self.ui.pushButton_1.clicked.connect(self.choose_input_json_dir)
        self.ui.pushButton_2.clicked.connect(self.choose_output_txt_dir)
        self.ui.pushButton_start_1.clicked.connect(self.start_json2txt_conversion)

        self.ui.pushButton_3.clicked.connect(self.choose_input_image_dir)
        self.ui.pushButton_4.clicked.connect(self.choose_input_txt_dir)
        self.ui.pushButton_5.clicked.connect(self.choose_output_json_dir)
        self.ui.pushButton_start_2.clicked.connect(self.start_txt2json_conversion)

        self.ui.pushButton_quit.clicked.connect(self.close_ui)

    def open_classes_dialog(self):
        if self.classes_manager is None:
            self.classes_manager = Classes_ui()
            # self.classes_dialog.setWindowModality(Qt.ApplicationModal)
            self.classes_manager.result_ready.connect(self._on_classes_result)
        self.classes_manager.ui.exec() # 显示子窗口

    def _on_classes_result(self, result_list):
        if result_list:
            # self.ui.textEdit.setText(str(result_list))
            self.ui.lineEdit.setText(str(result_list))
        else:
            # self.ui.textEdit.clear()
            self.ui.lineEdit.clear()
        print("result_list:", result_list)

    def update_result(self, message):
        self.ui.textEdit_2.append(message)

    def choose_input_json_dir(self):
        dlg = QFileDialog(self, "选择输入 json 文件")  # 标题
        dlg.setWindowIcon(QIcon("icon/json-file.svg"))
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
            self.input_json_dir = folder1
            self.ui.lineEdit_1.setText(folder1)

    def choose_output_txt_dir(self):
        dlg = QFileDialog(self, "选择输出 TXT 文件目录")
        dlg.setWindowIcon(QIcon("icon/txt-folder.svg"))
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
            self.output_txt_dir = folder2
            self.ui.lineEdit_2.setText(folder2)

    def choose_input_image_dir(self):
        dlg = QFileDialog(self, "选择输入图片文件夹")
        dlg.setWindowIcon(QIcon("icon/image-folder.svg"))
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
            self.input_image_dir = dlg.selectedFiles()[0]
            self.ui.lineEdit_3.setText(self.input_image_dir)

    def choose_input_txt_dir(self):
        dlg = QFileDialog(self, "选择输入 TXT 文件目录")
        dlg.setWindowIcon(QIcon("icon/txt-folder.svg"))
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, False)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setNameFilter("TXT文件 (*.txt)")

        dlg.setLabelText(QFileDialog.DialogLabel.Accept, "确认")
        dlg.setLabelText(QFileDialog.DialogLabel.Reject, "取消")
        dlg.setLabelText(QFileDialog.DialogLabel.LookIn, "查看：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileName, "名称：")
        dlg.setLabelText(QFileDialog.DialogLabel.FileType, "类型：")

        if dlg.exec():
            self.input_txt_dir = dlg.selectedFiles()[0]
            self.ui.lineEdit_4.setText(self.input_txt_dir)

    def choose_output_json_dir(self):
        if not self.input_image_dir or not self.input_txt_dir:
            QMessageBox.warning(self, "错误", "请先选择图像路径和txt路径")
            return

        dlg = QFileDialog(self, "选择保存 json 位置")
        dlg.setWindowIcon(QIcon("icon/json-file.svg"))
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
            self.output_json_dir = dlg.selectedFiles()[0]
            self.ui.lineEdit_5.setText(self.output_json_dir)


    def start_txt2json_conversion(self):

        text = self.ui.lineEdit.text().strip()

        if not text:
            QMessageBox.warning(self, "错误", "请先选择类别")
            return

        try:
            # chosen = eval(text)
            chosen = ast.literal_eval(text)  # 安全转换
            if not isinstance(chosen, list) or not chosen:
                raise ValueError

        except Exception:
            QMessageBox.critical(self, "错误", "无效的类别列表，请重新选择类别")
            return

        if not self.input_image_dir or not self.input_txt_dir or not self.output_json_dir:
            QMessageBox.warning(self, "错误", "请选择输入和输出目录")
            return

        self.ui.textEdit_2.clear()

        convert_yolo_to_coco(images_dir=self.input_image_dir,
                             labels_dir=self.input_txt_dir,
                             output_path=self.output_json_dir,
                             classes_list=chosen,
                             update_callback=self.update_result)

        QMessageBox.information(self, "完成", "所有文件转换完成")


    def start_json2txt_conversion(self):

        if not self.input_json_dir or not self.output_txt_dir:
            QMessageBox.warning(self, "错误", "请选择输入和输出目录")
            return

        self.ui.textEdit_2.clear()

        convert_coco_to_yolo(self.input_json_dir, self.output_txt_dir, self.update_result)

        QMessageBox.information(self, "完成", "所有文件转换完成")

    def close_ui(self):
        self.ui.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = COCO_and_YOLO()
    window.ui.show()
    sys.exit(app.exec())

