from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox
from PySide6.QtCore import QThread, Signal, QMutex
import sys
from deskapp.dashboard import Dashboard
from deskapp.newrun import NewRun
from deskapp.results import Results
from demonstrator.backend.loaders import name_to_runnable, dataset_loaders, model_loaders, parameters_to_class, analysis_methods
import traceback

items = list()


class DatasetLoaderThread(QThread):
    finished_success = Signal(object)
    finished_failure = Signal(str)
    canceled = Signal()

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self._is_canceled = False
        self.mutex = QMutex()

    def run(self):
        try:
            self.mutex.lock()
            if self._is_canceled:
                self.mutex.unlock()
                self.canceled.emit()
                return
            self.mutex.unlock()

            self.pipeline["dataset"]["return"] = name_to_runnable[self.pipeline["dataset"]["module"]](**self.pipeline["dataset"]["params"])

            self.mutex.lock()
            if self._is_canceled:
                self.mutex.unlock()
                self.canceled.emit()
                return
            self.mutex.unlock()

            self.finished_success.emit(self.pipeline)
        except Exception as e:
            traceback.print_exception(e)
            if not self._is_canceled:
                self.pipeline["status"] = "failed"
                traceback.print_exception(e)
                self.finished_failure.emit(str(e))

    def cancel(self):
        self.mutex.lock()
        self._is_canceled = True
        self.mutex.unlock()


class SelectDataset(NewRun):
    def next(self):
        self.save("dataset")

        self.loading_message = QMessageBox(self)
        self.loading_message.setWindowTitle("Loading Dataset")
        self.loading_message.setText("Please wait while the dataset is loading...")
        self.loading_message.setStandardButtons(QMessageBox.Cancel)
        self.loading_message.setModal(True)
        self.loading_message.button(QMessageBox.Cancel).clicked.connect(self.cancel_loading)
        self.loading_message.show()

        self.thread = DatasetLoaderThread(self.runs[-1])
        self.thread.finished_success.connect(self.on_success)
        self.thread.finished_failure.connect(self.on_failure)
        self.thread.canceled.connect(self.on_cancel)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_success(self, pipeline):
        self.loading_message.done(0)
        self.stacked_widget.setCurrentIndex(2)

    def on_failure(self, error_message):
        self.loading_message.done(0)
        self.show_error_message(error_message)

    def on_cancel(self):
        self.loading_message.done(0)

    def cancel_loading(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.cancel()
            self.thread.wait()  # Wait until the thread finishes
            self.loading_message.done(0)

    def switch_to_dashboard(self):
        self.save("dataset")
        self.runs[-1]["status"] = "saved"
        self.stacked_widget.setCurrentIndex(0)

    def closeEvent(self, event):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.cancel()
            self.thread.wait()  # Ensure the thread has finished
        event.accept()

class ModelLoaderThread(QThread):
    finished_success = Signal(object)
    finished_failure = Signal(str)
    canceled = Signal()

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self._is_canceled = False
        self.mutex = QMutex()

    def run(self):
        try:
            self.mutex.lock()
            if self._is_canceled:
                self.mutex.unlock()
                self.canceled.emit()
                return
            self.mutex.unlock()
            self.pipeline["model"]["return"] = name_to_runnable[self.pipeline["model"]["module"]](**self.pipeline["model"]["params"])
            self.mutex.lock()
            if self._is_canceled:
                self.mutex.unlock()
                self.canceled.emit()
                return
            self.mutex.unlock()
            self.finished_success.emit(self.pipeline)
        except Exception as e:
            if not self._is_canceled:
                self.pipeline["status"] = "failed"
                self.finished_failure.emit(str(e))

    def cancel(self):
        self.mutex.lock()
        self._is_canceled = True
        self.mutex.unlock()



class SelectModel(NewRun):
    def showEvent(self, event):
        pipeline = self.runs[-1]
        module = pipeline["dataset"]["module"]
        loaders = [loader for loader, values in model_loaders.items() if module in values["compatible"]]
        super().showEvent(event)
        self.dataset_selector.clear()
        self.dataset_selector.addItems(["Select a model loader"] + loaders)
        self.update_param_form("Select a model loader")

    def next(self):
        self.save("model")
        pipeline = self.runs[-1]

        self.loading_message = QMessageBox(self)
        self.loading_message.setWindowTitle("Loading Model")
        self.loading_message.setText("Please wait while the model is loading...")
        self.loading_message.setStandardButtons(QMessageBox.Cancel)
        self.loading_message.setModal(True)
        self.loading_message.button(QMessageBox.Cancel).clicked.connect(self.cancel_loading)
        self.loading_message.show()

        # Start the model loading thread using QThread
        self.thread = ModelLoaderThread(pipeline)
        self.thread.finished_success.connect(self.on_success)
        self.thread.finished_failure.connect(self.on_failure)
        self.thread.canceled.connect(self.on_cancel)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_success(self, pipeline):
        self.loading_message.done(0)
        self.stacked_widget.setCurrentIndex(3)

    def on_failure(self, error_message):
        self.loading_message.done(0)
        self.show_error_message(error_message)

    def on_cancel(self):
        self.loading_message.done(0)

    def cancel_loading(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.cancel()
            self.thread.wait()  # Ensure the thread is completely finished
            self.loading_message.done(0)

    def switch_to_dashboard(self):
        self.save("model")
        self.runs[-1]["status"] = "saved"
        self.stacked_widget.setCurrentIndex(0)

    def closeEvent(self, event):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.cancel()
            self.thread.wait()
        event.accept()

class AnalysisThread(QThread):
    finished_success = Signal(object)
    finished_failure = Signal(str)
    canceled = Signal()

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self._is_canceled = False
        self.mutex = QMutex()

    def run(self):
        try:
            self.mutex.lock()
            if self._is_canceled:
                self.mutex.unlock()
                self.canceled.emit()
                return
            self.mutex.unlock()
            args = self.pipeline["analysis"]["params"]
            dataset = self.pipeline["dataset"]["return"]
            model = self.pipeline["model"]["return"]
            sensitive = args["sensitive"]
            if "," in sensitive: sensitive = sensitive.split(",")
            else: sensitive = [sensitive]
            sensitive = [s.strip() for s in sensitive]
            args = {k:v for k,v in args.items() if k not in ["dataset", "model", "sensitive"]}
            self.pipeline["analysis"]["return"] = name_to_runnable[self.pipeline["analysis"]["module"]](dataset, model, sensitive, **args).text()
            self.pipeline["dataset"]["return"] = None
            self.pipeline["model"]["return"] = None
            self.pipeline["status"] = "completed"

            self.mutex.lock()
            if self._is_canceled:
                self.mutex.unlock()
                self.canceled.emit()
                return
            self.mutex.unlock()

            self.finished_success.emit(self.pipeline)
        except Exception as e:
            traceback.print_exception(e)
            if not self._is_canceled:
                self.pipeline["status"] = "failed"
                self.finished_failure.emit(str(e))

    def cancel(self):
        self.mutex.lock()
        self._is_canceled = True
        self.mutex.unlock()


class SelectAnalysis(NewRun):
    def showEvent(self, event):
        pipeline = self.runs[-1]
        compatible_methods = [
            method
            for method, entries in analysis_methods.items()
            if issubclass(
                parameters_to_class[pipeline["dataset"]["module"]]["return"],
                parameters_to_class[method][entries["parameters"][0][0]],
            )
            and issubclass(
                parameters_to_class[pipeline["model"]["module"]]["return"],
                parameters_to_class[method][entries["parameters"][1][0]],
            )
        ]
        super().showEvent(event)
        self.dataset_selector.clear()
        self.dataset_selector.addItems(["Select a fairness analysis method"] + compatible_methods)
        self.update_param_form("Select a fairness analysis method")

    def next(self):
        self.save("analysis")
        pipeline = self.runs[-1]

        self.loading_message = QMessageBox(self)
        self.loading_message.setWindowTitle("Running Fairness Analysis")
        self.loading_message.setText("Please wait while the fairness analysis is running...")
        self.loading_message.setStandardButtons(QMessageBox.Cancel)
        self.loading_message.setModal(True)
        self.loading_message.button(QMessageBox.Cancel).clicked.connect(self.cancel_loading)
        self.loading_message.show()

        # Start the analysis thread
        self.thread = AnalysisThread(pipeline)
        self.thread.finished_success.connect(self.on_success)
        self.thread.finished_failure.connect(self.on_failure)
        self.thread.canceled.connect(self.on_cancel)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_success(self, pipeline):
        self.loading_message.done(0)
        self.stacked_widget.setCurrentIndex(4)

    def on_failure(self, error_message):
        self.loading_message.done(0)
        self.show_error_message(error_message)

    def on_cancel(self):
        self.loading_message.done(0)

    def cancel_loading(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.cancel()
            self.thread.wait()  # Ensure thread is properly finished
            self.loading_message.done(0)

    def switch_to_dashboard(self):
        self.save("analysis")
        self.runs[-1]["status"] = "saved"
        self.stacked_widget.setCurrentIndex(0)

    def closeEvent(self, event):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.cancel()
            self.thread.wait()
        event.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MAMMOth Direct")
        self.setGeometry(100, 100, 1200, 800)
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(Dashboard(self.stacked_widget, items,
            {key: "<h1>"+key+"</h1>"+module["description"] for key, module in (dataset_loaders | model_loaders | analysis_methods).items()}))
        self.stacked_widget.addWidget(SelectDataset(self.stacked_widget,  dataset_loaders, items))
        self.stacked_widget.addWidget(SelectModel(self.stacked_widget, model_loaders, items))
        self.stacked_widget.addWidget(SelectAnalysis(self.stacked_widget, analysis_methods, items))
        self.stacked_widget.addWidget(Results(self.stacked_widget, items))
        self.setCentralWidget(self.stacked_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
