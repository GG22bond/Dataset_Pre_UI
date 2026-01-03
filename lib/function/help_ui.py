import sys

from PySide6.QtCore import QFile, QSize
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication, QMessageBox,
    QInputDialog, QTableWidgetItem, QDialog
)
from PySide6.QtGui import QIcon


class help_UI(QDialog):
    def __init__(self):
        super(help_UI, self).__init__()

        qfile = QFile("ui/help.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()
        self.help_ui = QUiLoader().load(qfile)
        if not self.help_ui:
            print(QUiLoader().errorString())
            sys.exit(-1)

        self.help_ui.setWindowIcon(QIcon("icon/help.svg"))

        self.help_ui.pushButton_1.setIcon(QIcon("icon/GitHub-logo.svg"))
        self.help_ui.pushButton_1.setIconSize(QSize(30, 30))
        self.help_ui.pushButton_2.setIcon(QIcon("icon/GitCode-logo.svg"))
        self.help_ui.pushButton_2.setIconSize(QSize(25, 25))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = help_UI()
    window.help_ui.show()
    sys.exit(app.exec())