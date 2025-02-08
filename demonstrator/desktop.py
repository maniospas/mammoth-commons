from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
import sys
from deskapp.dashboard import Dashboard
from deskapp.newrun import NewRun
from demonstrator.backend.loaders import name_to_runnable, dataset_loaders, model_loaders, parameters_to_class, analysis_methods

items = list()

class SelectModel(NewRun):
    def showEvent(self, event):
        pipeline = self.runs[-1]
        module = pipeline["dataset"]["module"]
        loaders = [loader for loader, values in model_loaders.items() if module in values["compatible"]]
        super().showEvent(event)
        self.dataset_selector.clear()
        self.dataset_selector.addItems(["Please select a model loader"] + loaders)
        self.update_param_form("Please select a model loader")

    def next(self):
        self.save("model")
        self.stacked_widget.setCurrentIndex(0)

    def switch_to_dashboard(self):
        self.save("model")
        self.stacked_widget.setCurrentIndex(0)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MAMMOth Direct Solutions")
        self.setGeometry(100, 100, 1200, 800)
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(Dashboard(self.stacked_widget, items, {module["name"]: module["description"] for module in dataset_loaders.values()}))
        self.stacked_widget.addWidget(NewRun(self.stacked_widget,  dataset_loaders, items))
        self.stacked_widget.addWidget(SelectModel(self.stacked_widget, model_loaders, items))
        self.setCentralWidget(self.stacked_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
