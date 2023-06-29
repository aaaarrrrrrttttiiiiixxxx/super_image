import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QWidget, QVBoxLayout, QHBoxLayout, \
    QScrollArea, QFileDialog, QPushButton, QComboBox
from PyQt5.QtGui import QPixmap, QPainter, QTransform, QPalette
from PyQt5.QtCore import Qt, QFile, QTextStream
from super_image import MsrnModel, MsrnConfig
from super_image.models.a2n.configuration_a2n import A2nConfig
from super_image.models.a2n.modeling_a2n import A2nModel
from super_image.models.carn.configuration_carn import CarnConfig
from super_image.models.carn.modeling_carn import CarnModel
from super_image.models.edsr.configuration_edsr import EdsrConfig
from super_image.models.edsr.modeling_edsr import EdsrModel
from super_image.models.pan.configuration_pan import PanConfig
from super_image.models.pan.modeling_pan import PanModel
import torch
from srgan import srgan_upscale
from srgan import esrgan_upscale


# GraphicsView для отображения
class CustomGraphicsView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self.scale_factor = 1.15

    def wheelEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            view_pos = event.pos()
            scene_pos = self.mapToScene(view_pos)
            old_transform = self.transform()

            old_scale = old_transform.m11()
            new_scale = old_scale * self.scale_factor if event.angleDelta().y() > 0 else old_scale / self.scale_factor

            zoom_transform = QTransform().scale(new_scale / old_scale, new_scale / old_scale)
            new_transform = old_transform * zoom_transform

            self.setTransform(new_transform)

            self.centerOn(scene_pos)
        else:
            super().wheelEvent(event)
        self.setDragMode(QGraphicsView.ScrollHandDrag)


class MyWindow(QMainWindow):
    # инициализация всех виджитов
    def __init__(self):
        super().__init__()
        self.setup_styles()

        self.original_scene = QGraphicsScene(self)
        self.original_view = CustomGraphicsView(self.original_scene)

        self.generated_scene = QGraphicsScene(self)
        self.generated_view = CustomGraphicsView(self.generated_scene)

        self.central_widget = QWidget()

        original_scroll_area = QScrollArea()
        original_scroll_area.setWidgetResizable(True)
        original_scroll_area.setWidget(self.original_view)

        generated_scroll_area = QScrollArea()
        generated_scroll_area.setWidgetResizable(True)
        generated_scroll_area.setWidget(self.generated_view)

        button_layout = QVBoxLayout()

        button_load = QPushButton()
        button_load.clicked.connect(self.loadFile)
        button_load.setText("Load image")
        button_load.setFixedSize(100, 50)
        button_layout.addWidget(button_load)

        self.comboBox = QComboBox()
        self.comboBox.addItems(['CARN', 'PAN', 'A2N', 'EDSR', 'MSRN', 'SRGAN', 'ESRGAN'])
        self.comboBox.setFixedSize(100, 50)
        button_layout.addWidget(self.comboBox)

        button_go = QPushButton()
        button_go.clicked.connect(self.go)
        button_go.setText("Upscale x4")
        button_go.setFixedSize(100, 50)
        button_layout.addWidget(button_go)

        button_save = QPushButton()
        button_save.clicked.connect(self.save)
        button_save.setText("Save image")
        button_save.setFixedSize(100, 50)
        button_layout.addWidget(button_save)

        layout = QHBoxLayout(self.central_widget)
        layout.addWidget(original_scroll_area)
        layout.addLayout(button_layout)
        layout.addWidget(generated_scroll_area)

        self.setCentralWidget(self.central_widget)

        self.original_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.generated_view.setDragMode(QGraphicsView.ScrollHandDrag)

        self.original_view.setRenderHint(QPainter.Antialiasing)
        self.generated_view.setRenderHint(QPainter.Antialiasing)
        self.in_filename = None

    def loadFile(self):
        self.in_filename, _ = QFileDialog.getOpenFileName(None, 'Open File', '.', 'Image files (*.jpg *.gif *.png)')
        if self.in_filename:
            self.original_scene.clear()
            original_pixmap = QPixmap(self.in_filename)
            self.original_scene.addPixmap(original_pixmap)

    def go(self):
        model_name = self.comboBox.currentText()
        if model_name == 'SRGAN':
            cv2.imwrite('images/hr.png', cv2.cvtColor(srgan_upscale(self.in_filename) * 255, cv2.COLOR_BGR2RGB))
        if model_name == 'ESRGAN':
            cv2.imwrite('images/hr.png', cv2.cvtColor(esrgan_upscale(self.in_filename) * 255, cv2.COLOR_BGR2RGB))
        else:
            cv2.imwrite('images/hr.png', cv2.cvtColor(self.upscale(self.in_filename, self.getModel()) * 255, cv2.COLOR_BGR2RGB))
        self.generated_scene.clear()
        generated_pixmap = QPixmap("images/hr.png")
        self.generated_scene.addPixmap(generated_pixmap)

    def getModel(self):
        model_name = self.comboBox.currentText()
        if model_name == 'MSRN':
            model = MsrnModel(MsrnConfig(scale=4, bam=True,))
            torch_loaded_model = torch.load('models/msnr.pth', map_location=torch.device('cpu'))
            model.load_state_dict(torch_loaded_model, False)
        elif model_name == 'EDSR':
            model = EdsrModel(EdsrConfig(scale=4))
            torch_loaded_model = torch.load('models/edsr.pth', map_location=torch.device('cpu'))
            model.load_state_dict(torch_loaded_model)
        elif model_name == 'A2N':
            model = A2nModel(A2nConfig(scale=4))
            torch_loaded_model = torch.load('models/a2n.pth', map_location=torch.device('cpu'))
            model.load_state_dict(torch_loaded_model)
        elif model_name == 'PAN':
            model = PanModel(PanConfig(scale=4, bam=True,))
            torch_loaded_model = torch.load('models/pan.pth', map_location=torch.device('cpu'))
            model.load_state_dict(torch_loaded_model)
        elif model_name == 'CARN':
            model = CarnModel(CarnConfig(scale=4))
            torch_loaded_model = torch.load('models/carn.pth', map_location=torch.device('cpu'))
            model.load_state_dict(torch_loaded_model)
        return model

    # метод улучшения при помощи модели из super_image
    @staticmethod
    def upscale(filename, model):
        lr = cv2.imread(filename)
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        low_res = torch.as_tensor(lr, dtype=torch.float).permute(2, 0, 1) / 255
        model.eval()
        high_res_fake = model.forward(low_res[None, :])[0]
        high_res_fake = high_res_fake.permute(1, 2, 0).detach().numpy()
        return high_res_fake

    def save(self):
        img = cv2.imread("images/hr.png")

        foldername = QFileDialog.getExistingDirectory(None, "Выберите папку для сохранения")
        if foldername != '':
            cv2.imwrite(foldername + '/hr.png', img)

    def setup_styles(self):
        style_file = QFile("scratch.css")
        style_file.open(QFile.ReadOnly | QFile.Text)
        stream = QTextStream(style_file)
        style = stream.readAll()
        style_file.close()
        QApplication.instance().setStyleSheet(style)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    window.setGeometry(300, 300, 1000, 500)
    app.exec()
