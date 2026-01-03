from PySide6.QtCore import Qt, QRectF, QSize
from PySide6.QtGui import QPixmap, QPainter, QWheelEvent

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsItem
)

class ViewImageWindow(QGraphicsView):
    '''
        From: https://www.cnblogs.com/zhiyiYo/p/15676079.html. Thank You!
    '''

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.zoomInTimes = 0
        self.maxZoomInTimes = 22  # 经验值，大约能够放大到原图的1.1^22≈8倍

        self.scene = QGraphicsScene(self)
        self.pixmapItem = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmapItem)
        self.setScene(self.scene)

        self.displayedImageSize = QSize(0, 0)

        # 隐藏滚动条
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 缩放围绕点(锚点)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        # 平滑缩放
        self.pixmapItem.setTransformationMode(Qt.SmoothTransformation)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        # 初始大小， 最小限制
        self.resize(400, 300)
        self.setMinimumSize(100, 75)

        super().fitInView(self.pixmapItem, Qt.KeepAspectRatio)
        self.zoomInTimes = 0

    # 滚轮缩放
    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            self.zoomIn()
        else:
            self.zoomOut()

    # 窗口尺寸缩放
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.zoomInTimes > 0:
            return
        ratio = self.__getScaleRatio()
        self.displayedImageSize = self.pixmapItem.pixmap().size() * ratio
        if ratio < 1:
            super().fitInView(self.pixmapItem, Qt.KeepAspectRatio)
        else:
            super().resetTransform()

    # 刷新图片
    def setImage(self, pixmap: QPixmap):
        super().resetTransform()
        self.zoomInTimes = 0
        self.__setDragEnabled(False)

        self.pixmapItem.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))

        ratio = self.__getScaleRatio()
        self.displayedImageSize = pixmap.size() * ratio
        if ratio < 1:
            super().fitInView(self.pixmapItem, Qt.KeepAspectRatio)

    # 重置变换
    def resetTransform(self):
        super().resetTransform()
        self.zoomInTimes = 0
        self.__setDragEnabled(False)

    # 根据图片的尺寸判断是否允许拖拽
    def __isEnableDrag(self):
        v = self.verticalScrollBar().maximum() > 0
        h = self.horizontalScrollBar().maximum() > 0
        return v or h

    # 启用/禁用拖拽
    def __setDragEnabled(self, enabled: bool):

        mode = QGraphicsView.ScrollHandDrag if enabled else QGraphicsView.NoDrag
        self.setDragMode(mode)

    # 计算显示的图像和原始图像的缩放比例
    def __getScaleRatio(self):
        pixmap = self.pixmapItem.pixmap()
        if pixmap.isNull():
            return 1
        pw = pixmap.width()
        ph = pixmap.height()
        vw = self.viewport().width()
        vh = self.viewport().height()
        rw = min(1, vw / pw)
        rh = min(1, vh / ph)
        return min(rw, rh)

    # 缩放场景使其适应窗口大小
    def fitInView(self, item: QGraphicsItem, mode=Qt.KeepAspectRatio):
        super().fitInView(item, mode)
        self.displayedImageSize = self.__getScaleRatio() * self.pixmapItem.pixmap().size()
        self.zoomInTimes = 0

    # 放大
    def zoomIn(self, anchor=QGraphicsView.AnchorUnderMouse):
        if self.zoomInTimes == self.maxZoomInTimes:
            return
        self.setTransformationAnchor(anchor)
        self.zoomInTimes += 1
        self.scale(1.1, 1.1)
        self.__setDragEnabled(self.__isEnableDrag())
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    # 缩小
    def zoomOut(self, anchor=QGraphicsView.AnchorUnderMouse):
        if self.zoomInTimes == 0 and not self.__isEnableDrag():
            return
        self.setTransformationAnchor(anchor)
        self.zoomInTimes -= 1

        pixmap = self.pixmapItem.pixmap()
        pw = pixmap.width()
        ph = pixmap.height()
        w = self.displayedImageSize.width() * (1.1 ** self.zoomInTimes)
        h = self.displayedImageSize.height() * (1.1 ** self.zoomInTimes)

        vw = self.viewport().width()
        vh = self.viewport().height()

        if pw > vw or ph > vh:
            if w <= vw and h <= vh:
                super().fitInView(self.pixmapItem, Qt.KeepAspectRatio)
            else:
                self.scale(1 / 1.1, 1 / 1.1)
        else:
            if w <= pw:
                super().resetTransform()
                self.zoomInTimes = 0
                self.__setDragEnabled(False)
            else:
                self.scale(1 / 1.1, 1 / 1.1)

        self.__setDragEnabled(self.__isEnableDrag())
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)