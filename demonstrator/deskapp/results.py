from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSizePolicy
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWebEngineWidgets import QWebEngineView

def format_run(run):
    return "[" + run["timestamp"] + "] " + run["description"]

class Results(QWidget):
    def __init__(self, stacked_widget, runs):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.runs = runs

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Create button layout first (buttons at the top)
        button_layout = QHBoxLayout()

        self.edit_button = QPushButton("Edit", self)
        self.edit_button.setStyleSheet("background-color: #17a2b8; color: white; padding: 6px; border-radius: 5px;")
        self.edit_button.clicked.connect(self.edit_run)
        button_layout.addWidget(self.edit_button)

        self.variation_button = QPushButton("Create variation", self)
        self.variation_button.setStyleSheet(
            "background-color: #ffc107; color: black; padding: 6px; border-radius: 5px;")
        self.variation_button.clicked.connect(self.create_variation)
        button_layout.addWidget(self.variation_button)

        self.delete_button = QPushButton("Delete", self)
        self.delete_button.setStyleSheet("background-color: #dc3545; color: white; padding: 6px; border-radius: 5px;")
        self.delete_button.clicked.connect(self.delete_run)
        button_layout.addWidget(self.delete_button)

        self.close_button = QPushButton("Close", self)
        self.close_button.setStyleSheet("background-color: #6c757d; color: white; padding: 6px; border-radius: 5px;")
        self.close_button.clicked.connect(self.switch_to_dashboard)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)  # Add buttons first

        # Title label
        self.title_label = QLabel("Analysis outcome", self)
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self.title_label)

        # Results Viewer (Stretches to fill available space)
        self.results_viewer = QWebEngineView(self)
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.results_viewer.setSizePolicy(size_policy)
        layout.addWidget(self.results_viewer, 1)  # "1" ensures it stretches in available space

        self.setLayout(layout)

    def switch_to_dashboard(self):
        self.stacked_widget.setCurrentIndex(0)

    def showEvent(self, event):
        super().showEvent(event)
        if self.runs:
            self.title_label.setText(format_run(self.runs[-1]))
            html_content = self.runs[-1]["analysis"].get("return", "<p>No results available.</p>")
        else:
            html_content = "<p>No results available.</p>"

        # Use QTimer to ensure the WebEngineView renders properly
        QTimer.singleShot(100, lambda: self.results_viewer.setHtml(html_content))
        self.results_viewer.show()

    def edit_run(self):
        if self.runs:
            self.stacked_widget.setCurrentIndex(1)

    def create_variation(self):
        if not self.runs:
            return
        new_run = self.runs[-1].copy()
        new_run["status"] = "new"
        self.runs.append(new_run)
        self.stacked_widget.setCurrentIndex(1)

    def delete_run(self):
        if not self.runs:
            return
        self.runs.pop()
        self.stacked_widget.setCurrentIndex(0)  # Go back to the dashboard
