import sys
from PySide6.QtCore import QFile, Qt, Signal
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication, QMessageBox,
    QInputDialog, QTableWidgetItem, QDialog
)
from PySide6.QtGui import QIcon
import ast

class Classes_ui(QDialog):

    result_ready = Signal(list)

    def __init__(self):
        super(Classes_ui, self).__init__()

        qfile = QFile("ui/classes_manage.ui")
        if not qfile.open(QFile.ReadOnly):
            QMessageBox.critical(self, '错误', '无法打开 UI 文件')
            sys.exit(-1)
        self.ui = QUiLoader().load(qfile)
        qfile.close()
        if not self.ui:
            QMessageBox.critical(self, '错误', QUiLoader().errorString())
            sys.exit(-1)

        # self.ui.tableWidget.setColumnCount(2)
        # self.ui.tableWidget.verticalHeader().setVisible(False)

        self.ui.setWindowIcon(QIcon("icon/DL-logo.svg"))

        self.ui.pushButton_1.clicked.connect(self.add_row)
        self.ui.pushButton_2.clicked.connect(self.remove_row)
        self.ui.pushButton_4.clicked.connect(self.confirm_and_close)
        self.ui.pushButton_5.clicked.connect(self.confirm_and_close_and_close)

        self.result = []

        self.show_placeholder()
        self.sync_to_textedit()

    def show_placeholder(self):

        if self.ui.tableWidget.rowCount() == 0:
            self.ui.tableWidget.insertRow(0)
            # 第一列提示“序号”
            num_ph = QTableWidgetItem('序号')
            # 不可编辑、不可选中
            num_ph.setFlags(Qt.NoItemFlags)
            self.ui.tableWidget.setItem(0, 0, num_ph)
            # 第二列提示文字
            label_ph = QTableWidgetItem('点击 “+” 添加，点击 “-” 删除')
            label_ph.setFlags(Qt.NoItemFlags)
            self.ui.tableWidget.setItem(0, 1, label_ph)

    def remove_placeholder(self):

        if self.ui.tableWidget.rowCount() == 1:
            item = self.ui.tableWidget.item(0, 0)

            if item and (item.flags() == Qt.NoItemFlags):
                self.ui.tableWidget.removeRow(0)

    def sync_to_textedit(self):

        labels = []
        for i in range(self.ui.tableWidget.rowCount()):
            item = self.ui.tableWidget.item(i, 1)
            if item and item.flags() & Qt.ItemIsEditable:
                text = item.text().strip()
                if text:
                    labels.append(text)
        if labels:
            self.ui.lineEdit.setText(str(labels))
            # self.ui.textEdit.setText(str(labels))
        else:
            self.ui.lineEdit.clear()
            # self.ui.textEdit.clear()

    def add_row(self):

        self.remove_placeholder()
        row = self.ui.tableWidget.rowCount()
        self.ui.tableWidget.insertRow(row)
        # num 列，只读
        num_item = QTableWidgetItem(str(row))
        num_item.setFlags(num_item.flags() & ~Qt.ItemIsEditable)
        self.ui.tableWidget.setItem(row, 0, num_item)

        while True:
            text, ok = QInputDialog.getText(self, '添加类别', '请输入类别名称:')
            if not ok:
                self.ui.tableWidget.removeRow(row)
                self.show_placeholder()
                return
            name = text.strip()
            if not name:
                QMessageBox.warning(self, '提示', '类别名称不能为空！')
                continue
            existing = [
                self.ui.tableWidget.item(i, 1).text().strip()
                for i in range(self.ui.tableWidget.rowCount())
                if self.ui.tableWidget.item(i, 1) and (self.ui.tableWidget.item(i, 1).flags() & Qt.ItemIsEditable)
            ]
            if name in existing:
                QMessageBox.warning(self, '提示', '请勿输入相同类别！')
                continue
            break
        # 填充 label 列，可编辑
        label_item = QTableWidgetItem(name)
        label_item.setFlags(label_item.flags() | Qt.ItemIsEditable)
        self.ui.tableWidget.setItem(row, 1, label_item)
        self.sync_to_textedit()

    def remove_row(self):

        row = self.ui.tableWidget.currentRow()
        if row < 0:
            QMessageBox.warning(self, '提示', '请先选中要删除的行')
            return
        self.ui.tableWidget.removeRow(row)

        for i in range(self.ui.tableWidget.rowCount()):
            item = self.ui.tableWidget.item(i, 0)
            # num重新排序
            # if item and (item.flags() & Qt.ItemIsEditable) == 0:
            if item and item.flags() != Qt.NoItemFlags:
                item.setText(str(i))

        if self.ui.tableWidget.rowCount() == 0:
            self.show_placeholder()
        self.sync_to_textedit()

    def confirm_and_close(self):

        self.sync_to_textedit()
        # text = self.ui.textEdit.toPlainText().strip()
        text = self.ui.lineEdit.text().strip()
        if text:
            try:
                self.result = ast.literal_eval(text)
            except Exception:
                self.result = [
                    self.ui.tableWidget.item(i, 1).text().strip()
                    for i in range(self.ui.tableWidget.rowCount())
                    if self.ui.tableWidget.item(i, 1) and (self.ui.tableWidget.item(i, 1).flags() & Qt.ItemIsEditable)
                ]
        else:
            self.result = []
        print(self.result)
        self.result_ready.emit(self.result)

    def confirm_and_close_and_close(self):
        self.confirm_and_close()
        self.ui.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Classes_ui()
    window.ui.show()
    sys.exit(app.exec())

