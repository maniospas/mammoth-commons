from PySide6.QtWidgets import (
    QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout,
    QScrollArea, QMessageBox, QSizePolicy, QSpacerItem
)
from PySide6.QtCore import Qt
from datetime import datetime


class Dashboard(QWidget):
    def __init__(self, stacked_widget, runs, tag_descriptions):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.runs = runs

        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.label = QLabel("Dashboard", self)
        self.label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.main_layout.addWidget(self.label)

        new_button = QPushButton("New", self)
        new_button.setStyleSheet("background-color: #007bff; color: white; padding: 8px; border-radius: 5px;")
        new_button.clicked.connect(self.create_new_item)
        self.main_layout.addWidget(new_button)

        # Scroll Area (Fixed Size, but allows scrolling)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Content Widget inside Scroll Area
        self.content_widget = QWidget()
        self.layout = QVBoxLayout(self.content_widget)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.setSpacing(2)
        self.scroll_area.setWidget(self.content_widget)

        self.main_layout.addWidget(self.scroll_area)
        self.setLayout(self.main_layout)
        self.tag_descriptions = tag_descriptions

        self.refresh_dashboard()

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                elif child.layout():
                    self.clear_layout(child.layout())

    def refresh_dashboard(self):
        # Clear existing items
        self.clear_layout(self.layout)

        for index, run in enumerate(self.runs):
            item_layout = QHBoxLayout()

            description_label = QLabel(format_run(run), self)
            description_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

            spacer = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)

            tag_container = QHBoxLayout()
            tag_container.setAlignment(Qt.AlignmentFlag.AlignRight)

            tags = []
            if "dataset" in run: tags.append(run["dataset"]["module"])
            if "model" in run: tags.append(run["model"]["module"])
            if "analysis" in run: tags.append(run["analysis"]["module"])

            for tag in tags:
                tag_label = QPushButton(f" {tag} ", self)
                tag_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                tag_label.setStyleSheet("background-color: gray; color: white; padding: 2px 6px; border-radius: 10px;")
                tag_label.clicked.connect(lambda checked, t=tag: self.show_tag_description(t))
                tag_container.addWidget(tag_label)

            if run.get("status") == "complete":
                action_button = QPushButton("👁", self)
                action_button.setStyleSheet("background-color: #17a2b8; border-radius: 5px; color: white;")
                action_button.clicked.connect(lambda checked, i=index: self.view_result(i))
            else:
                action_button = QPushButton("✎", self)
                action_button.setStyleSheet("background-color: #ffc107; border-radius: 5px; color: white;")
                action_button.clicked.connect(lambda checked, i=index: self.edit_item(i))

            action_button.setFixedSize(30, 30)

            delete_button = QPushButton("🗑", self)
            delete_button.setStyleSheet("background-color: #dc3545; border-radius: 5px; color: white;")
            delete_button.setFixedSize(30, 30)
            delete_button.clicked.connect(lambda checked, i=index: self.delete_item(i))

            item_layout.addWidget(description_label)
            item_layout.addItem(spacer)
            item_layout.addLayout(tag_container)
            item_layout.addWidget(action_button)
            item_layout.addWidget(delete_button)

            self.layout.addLayout(item_layout)

        self.content_widget.adjustSize()

    def showEvent(self, event):
        self.refresh_dashboard()
        super().showEvent(event)

    def edit_item(self, index):
        # Move the run to the end before editing
        run = self.runs.pop(index)
        self.runs.append(run)
        self.refresh_dashboard()
        self.stacked_widget.setCurrentIndex(1)  # Navigate to editing page

    def view_result(self, index):
        # Move the completed run to the end before viewing results
        run = self.runs.pop(index)
        self.runs.append(run)
        self.refresh_dashboard()
        self.stacked_widget.setCurrentIndex(4)  # Navigate to results page

    def show_tag_description(self, tag):
        msg = QMessageBox()
        msg.setWindowTitle("Help")
        msg.setText(self.tag_descriptions.get(tag, "No description available."))
        msg.exec()

    def create_new_item(self):
        self.runs.append({
            "description": "Fairness analysis",
            "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M"),
            "status": "in_progress"
        })
        self.stacked_widget.setCurrentIndex(1)

    def delete_item(self, index):
        reply = QMessageBox.question(self, "Delete?",
                                     f"Confirm the deletion of {format_run(self.runs[index])}.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.runs.pop(index)
            self.refresh_dashboard()

def format_run(run):
    return "[" + run["timestamp"] + "] " + run["description"]
