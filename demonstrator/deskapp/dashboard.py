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
                # If the item is a widget, delete it.
                if child.widget():
                    child.widget().deleteLater()
                # If the item is a layout, recursively clear it.
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

            if "result" in run:
                action_button = QPushButton("👁", self)
                action_button.setStyleSheet("background-color: #17a2b8; border-radius: 5px; color: white;")
                action_button.clicked.connect(lambda checked, res=run["result"]: self.view_result(res))
            else:
                action_button = QPushButton("✎", self)
                action_button.setStyleSheet("background-color: #ffc107; border-radius: 5px; color: white;")
                action_button.clicked.connect(lambda checked: self.edit_item(run["description"]))

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

        # Ensure scrolling works by letting the content widget grow
        self.content_widget.adjustSize()

    def showEvent(self, event):
        self.refresh_dashboard()
        super().showEvent(event)

    def edit_item(self, description):
        print(f"Editing item: {description}")

    def view_result(self, result):
        msg = QMessageBox()
        msg.setWindowTitle("View Result")
        msg.setText(str(result))
        msg.exec()

    def show_tag_description(self, tag):
        msg = QMessageBox()
        msg.setWindowTitle("Tag Description")
        msg.setText(self.tag_descriptions.get(tag, "No description available."))
        msg.exec()

    def create_new_item(self):
        self.runs.append({
            "description": "Fairness analysis",
            "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M")
        })
        self.stacked_widget.setCurrentIndex(1)

    def delete_item(self, index):
        reply = QMessageBox.question(self, "Delete?",
                                     f"Are you sure you want to delete {format_run(self.runs[index])}?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.runs.pop(index)
            self.refresh_dashboard()
            self.update()

def format_run(run):
    return "[" + run["timestamp"] + "] " + run["description"]