import sys
from PySide6.QtCore import QFile, Qt, QCoreApplication, QUrl
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox
)
from PySide6.QtGui import QIcon, QDesktopServices
from lib.VOC_and_YOLO_ui import VOC_and_YOLO
from lib.COCO_and_YOLO_ui import COCO_and_YOLO
from lib.COCO_and_VOC_ui import COCO_and_VOC
from lib.amp_main_ui import Amplification_ui
from lib.segmentation_ui import Segmentation_Ui
from lib.change_imagesize_ui import Change_imagesize_ui
from lib.style_transfer_ui import Style_transfer_ui
from lib.function.help_ui import help_UI

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        qfile = QFile("ui/main_ui.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()
        self.ui = QUiLoader().load(qfile)
        if not self.ui:
            print(QUiLoader().errorString())
            sys.exit(-1)

        self.VOC_and_YOLO_UI = None
        self.COCO_and_YOLO_UI = None
        self.COCO_and_VOC_UI = None
        self.amp_UI = None
        self.segmentation_UI = None
        self.change_imagesize_UI = None
        self.style_transfer_UI = None
        self.help_ui = None

        self.ui.setWindowIcon(QIcon("icon/DL-logo.svg"))

        self.ui.pushButton_VOC_to_YOLO.clicked.connect(self.open_VOC_and_YOLO_UI)
        self.ui.pushButton_YOLO_to_VOC.clicked.connect(self.open_VOC_and_YOLO_UI)
        self.ui.pushButton_COCO_to_YOLO.clicked.connect(self.open_COCO_and_YOLO_UI)
        self.ui.pushButton_YOLO_to_COCO.clicked.connect(self.open_COCO_and_YOLO_UI)
        self.ui.pushButton_COCO_to_VOC.clicked.connect(self.open_COCO_and_VOC_UI)
        self.ui.pushButton_VOC_to_COCO.clicked.connect(self.open_COCO_and_VOC_UI)
        self.ui.pushButton_amp.clicked.connect(self.open_amp_UI)
        self.ui.pushButton_COCO_to_Labelme.clicked.connect(self.open_segmentation_UI)
        self.ui.pushButton_Labelme_to_COCO.clicked.connect(self.open_segmentation_UI)
        self.ui.pushButton_COCO_to_YOLO_2.clicked.connect(self.open_segmentation_UI)
        self.ui.pushButton_change_size.clicked.connect(self.open_change_size_UI)
        self.ui.pushButton_style_trans.clicked.connect(self.open_style_transfer_UI)
        self.ui.pushButton_help.clicked.connect(self.open_help_dialog)

        self.ui.actionVOC_to_YOLO.triggered.connect(self.open_VOC_and_YOLO_UI)
        self.ui.actionYOLO_to_VOC.triggered.connect(self.open_VOC_and_YOLO_UI)
        self.ui.actionCOCO_to_YOLO.triggered.connect(self.open_COCO_and_YOLO_UI)
        self.ui.actionYOLO_to_COCO.triggered.connect(self.open_COCO_and_YOLO_UI)
        self.ui.actionCOCO_to_VOC.triggered.connect(self.open_COCO_and_VOC_UI)
        self.ui.actionVOC_to_COCO.triggered.connect(self.open_COCO_and_VOC_UI)
        self.ui.action_dataset_amp.triggered.connect(self.open_amp_UI)
        self.ui.actionCOCO_to_Labelme.triggered.connect(self.open_segmentation_UI)
        self.ui.actionCOOC_to_YOLO_seg.triggered.connect(self.open_segmentation_UI)
        self.ui.actionLabelme_to_COCO.triggered.connect(self.open_segmentation_UI)
        self.ui.action_change_image_size.triggered.connect(self.change_imagesize_UI)
        self.ui.action_style_transfer.triggered.connect(self.style_transfer_UI)

        self.ui.actionGithub.triggered.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/GG22bond/Dataset_Pre_UI")))
        self.ui.actionGitCode.triggered.connect(lambda: QDesktopServices.openUrl(QUrl("https://gitcode.com/dalidexiansen/Dataset_Pre_UI")))
        self.ui.action_quit.triggered.connect(self.close_ui)


    def open_help_dialog(self):
        if self.help_ui is None:
            self.help_ui = help_UI()
            self.help_ui.help_ui.pushButton_1.clicked.connect(
                lambda: QDesktopServices.openUrl(QUrl("https://github.com/GG22bond/Dataset_Pre_UI")))
            self.help_ui.help_ui.pushButton_2.clicked.connect(
                lambda: QDesktopServices.openUrl(QUrl("https://gitcode.com/dalidexiansen/Dataset_Pre_UI")))

        self.help_ui.help_ui.exec()

    def open_VOC_and_YOLO_UI(self):
        if self.VOC_and_YOLO_UI is None:
            self.VOC_and_YOLO_UI = VOC_and_YOLO()
        # 应用窗口模态
        self.VOC_and_YOLO_UI.ui.setWindowModality(Qt.ApplicationModal)
        self.VOC_and_YOLO_UI.ui.show()

    def open_COCO_and_YOLO_UI(self):
        if self.COCO_and_YOLO_UI is None:
            self.COCO_and_YOLO_UI = COCO_and_YOLO()
        self.COCO_and_YOLO_UI.ui.setWindowModality(Qt.ApplicationModal)
        self.COCO_and_YOLO_UI.ui.show()

    def open_COCO_and_VOC_UI(self):
        if self.COCO_and_VOC_UI is None:
            self.COCO_and_VOC_UI = COCO_and_VOC()
        self.COCO_and_VOC_UI.ui.setWindowModality(Qt.ApplicationModal)
        self.COCO_and_VOC_UI.ui.show()

    def open_amp_UI(self):
        if self.amp_UI is None:
            self.amp_UI = Amplification_ui()
        self.amp_UI.ui.setWindowModality(Qt.ApplicationModal)
        self.amp_UI.ui.show()

    def open_segmentation_UI(self):
        if self.segmentation_UI is None:
            self.segmentation_UI = Segmentation_Ui()
        self.segmentation_UI.ui.setWindowModality(Qt.ApplicationModal)
        self.segmentation_UI.ui.show()

    def open_change_size_UI(self):
        if self.change_imagesize_UI is None:
            self.change_imagesize_UI = Change_imagesize_ui()
        self.change_imagesize_UI.ui.setWindowModality(Qt.ApplicationModal)
        self.change_imagesize_UI.ui.show()

    def open_style_transfer_UI(self):
        if self.style_transfer_UI is None:
            self.style_transfer_UI = Style_transfer_ui()
        self.style_transfer_UI.ui.setWindowModality(Qt.ApplicationModal)
        self.style_transfer_UI.ui.show()

    def close_ui(self):
        self.ui.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.ui.show()
    sys.exit(app.exec())


