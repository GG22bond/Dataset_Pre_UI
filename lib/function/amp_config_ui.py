import sys
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Signal, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QDialog, QStyledItemDelegate, QLineEdit, QTableWidgetItem
from PySide6.QtGui import QDoubleValidator, QIntValidator

class NumericDelegate(QStyledItemDelegate):
    def __init__(self, parent, validators_map, defaults_map):
        super().__init__(parent)
        self.validators_map = validators_map
        self.defaults_map = defaults_map

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)

        font = editor.font()
        font.setPointSize(8)
        editor.setFont(font)

        row = index.row()
        validator = self.validators_map.get(row)
        default_val = self.defaults_map.get(row)

        if validator:
            editor.setValidator(validator)

            try:
                bottom = validator.bottom()
                top = validator.top()
                editor.setPlaceholderText(f"range: {bottom}-{top}, default: {default_val}")
            except Exception:
                pass
        else:
            default_validator = QIntValidator(editor)
            editor.setValidator(default_validator)
            try:
                bottom = default_validator.bottom()
                top = default_validator.top()
                editor.setPlaceholderText(f"range: {bottom}-{top}, default: {default_val}")
            except Exception:
                pass
        return editor

class Amp_config_ui(QDialog):

    result_ready = Signal(list)

    def __init__(self):
        super(Amp_config_ui, self).__init__()

        qfile = QFile("ui/amp_config_ui.ui")
        qfile.open(QFile.ReadOnly)
        qfile.close()
        self.ui = QUiLoader().load(qfile)
        if not self.ui:
            print(QUiLoader().errorString())
            sys.exit(-1)

        self.ui.tableWidget_1.setColumnWidth(0, 220)

        self.ui.setWindowIcon(QIcon("icon/config_ui.svg"))

        # 默认值：
        defaults = [0.01, 0.6, 30, 1, 0.3, 5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        ranges = [
            (0.0, 0.05, 3),  # 高斯噪声方差
            (0.55, 1.0, 2),  # 亮度 alpha
            (0.0, 50.0, 0),  # 擦除边长
            (0.0, 10.0, 0),  # 擦除个数
            (0.0, 0.5, 2),   # 擦除阈值
            (0.0, 60.0, 1),  # 旋转角度
            (0.0, 0.5, 2),  # 噪声处理概率
            (0.0, 0.5, 2),  # 亮度处理概率
            (0.0, 0.5, 2),  # 擦除处理概率
            (0.0, 0.5, 2),  # 旋转处理概率
            (0.0, 0.5, 2),  # 平移处理概率
            (0.0, 0.5, 2),  # 镜像处理概率
        ]

        for row, val in enumerate(defaults):
            item = QTableWidgetItem(str(val))
            self.ui.tableWidget_1.setItem(row, 1, item)

        validators_map = {}
        defaults_map = {}
        for row, (min_v, max_v, dec) in enumerate(ranges):
            v = QDoubleValidator(min_v, max_v, dec, self)
            v.setNotation(QDoubleValidator.StandardNotation)
            validators_map[row] = v
            defaults_map[row] = defaults[row]

        delegate = NumericDelegate(self.ui.tableWidget_1, validators_map, defaults_map)
        self.ui.tableWidget_1.setItemDelegateForColumn(1, delegate)

        # 第一列不可编辑
        for row in range(self.ui.tableWidget_1.rowCount()):
            item = self.ui.tableWidget_1.item(row, 0)
            if item:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

        self.ui.lineEdit.textChanged.connect(self.search_table)
        self.ui.pushButton_1.clicked.connect(self.emit_column_values)
        self.ui.pushButton_2.clicked.connect(self.close_event)

    def search_table(self):
        search_text = self.ui.lineEdit.text().lower()
        for row in range(self.ui.tableWidget_1.rowCount()):
            item0 = self.ui.tableWidget_1.item(row, 0)
            hidden = True
            if item0 and search_text in item0.text().lower():
                hidden = False
            self.ui.tableWidget_1.setRowHidden(row, hidden)

    def emit_column_values(self):
        values = []
        for row in range(self.ui.tableWidget_1.rowCount()):
            item = self.ui.tableWidget_1.item(row, 1)
            if item:
                text = item.text()
                if text:
                    try:
                        val = float(text) if '.' in text else int(text)
                    except ValueError:
                        continue
                    values.append(val)
        print(values)
        self.result_ready.emit(values)

    def close_event(self):
        self.emit_column_values()
        self.ui.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Amp_config_ui()
    window.ui.show()
    sys.exit(app.exec())
