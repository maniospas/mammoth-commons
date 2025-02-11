from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
import sys
from states.dashboard import Dashboard
from states.step import load_all_runs
from states.steps.dataset import SelectDataset
from states.steps.model import SelectModel
from states.steps.analysis import SelectAnalysis
from states.results import Results
from demonstrator.backend.loaders import dataset_loaders, model_loaders, analysis_methods

items = load_all_runs("history.json")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        tags = {key: "<h1>"+key+"</h1>"+module["description"] for key, module in (dataset_loaders | model_loaders | analysis_methods).items()}
        self.setWindowTitle("MAMMOth Direct")
        self.setGeometry(100, 100, 1200, 800)
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(Dashboard(self.stacked_widget, items, tags))
        self.stacked_widget.addWidget(SelectDataset("Data", self.stacked_widget,  dataset_loaders, items))
        self.stacked_widget.addWidget(SelectModel("Model", self.stacked_widget, model_loaders, items))
        self.stacked_widget.addWidget(SelectAnalysis("Analysis method", self.stacked_widget, analysis_methods, items))
        self.stacked_widget.addWidget(Results(self.stacked_widget, items, tags))
        self.setCentralWidget(self.stacked_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
