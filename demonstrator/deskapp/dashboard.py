from PySide6.QtWidgets import (
    QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout,
    QScrollArea, QMessageBox, QSizePolicy, QSpacerItem
)
from PySide6.QtCore import Qt
from datetime import datetime
from .newrun import save_all_runs


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

        new_button = self.create_icon_button("➕", "#007bff", "New analysis", self.create_new_item)
        self.main_layout.addWidget(new_button)

        # Scroll Area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Content Widget
        self.content_widget = QWidget()
        self.layout = QVBoxLayout(self.content_widget)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.setSpacing(2)
        self.scroll_area.setWidget(self.content_widget)

        self.main_layout.addWidget(self.scroll_area)
        self.setLayout(self.main_layout)
        self.tag_descriptions = tag_descriptions

        self.refresh_dashboard()

    def view_result(self, index):
        run = self.runs.pop(index)
        self.runs.append(run)
        self.refresh_dashboard()
        self.stacked_widget.setCurrentIndex(4)

    def edit_item(self, index):
        if not self.runs: return
        if self.runs[index].get("status", "") != "completed":
            reply = QMessageBox.Yes
        else:
            reply = QMessageBox.question(self, "Edit?",
                                         f"You can change modules and modify parameters of {format_run(self.runs[index])}. "
                                         "However, this will also remove its results. Consider creating a variation if you want to preserve current results.",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            run = self.runs.pop(index)
            self.runs.append(run)
            self.refresh_dashboard()
            self.stacked_widget.setCurrentIndex(1)

    def create_variation(self, index):
        if not self.runs:
            return
        new_run = self.runs[index].copy()
        new_run["status"] = "new"
        self.runs.append(new_run)
        self.refresh_dashboard()
        self.stacked_widget.setCurrentIndex(1)

    def create_new_item(self):
        self.runs.append({
            "description": "Fairness analysis",
            "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M"),
            "status": "in_progress"
        })
        self.stacked_widget.setCurrentIndex(1)
        self.refresh_dashboard()

    def delete_item(self, index):
        reply = QMessageBox.question(self, "Delete?",
                                     f"Confirm the deletion of {format_run(self.runs[index])}.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.runs.pop(index)
            self.refresh_dashboard()
            save_all_runs("history.json", self.runs)

    def create_icon_button(self, text, color, tooltip, callback):
        button = QPushButton(text, self)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color}; 
                color: white; 
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: {self.darken_color(color)};
            }}
        """)
        button.setFixedSize(30, 30)
        button.setToolTip(tooltip)
        button.clicked.connect(callback)
        return button

    def create_tag_button(self, text, tooltip, callback):
        button = QPushButton(text, self)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: gray; 
                color: white; 
                padding: 2px 6px; 
                border-radius: 10px;
            }}
            QPushButton:hover {{
                background-color: darkgray;
            }}
        """)
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        button.setToolTip(tooltip)
        button.clicked.connect(callback)
        return button

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                elif child.layout():
                    self.clear_layout(child.layout())


    def showEvent(self, event):
        self.refresh_dashboard()

    def refresh_dashboard(self):
        self.clear_layout(self.layout)

        for index, run in sorted(enumerate(self.runs), key=lambda x: x[1]["timestamp"]):
            item_layout = QHBoxLayout()
            item_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

            description_label = QLabel(format_run(run), self)
            description_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

            tag_container = QHBoxLayout()
            tag_container.setAlignment(Qt.AlignmentFlag.AlignLeft)

            tags = []
            if "dataset" in run: tags.append(run["dataset"]["module"])
            if "model" in run: tags.append(run["model"]["module"])
            if "analysis" in run: tags.append(run["analysis"]["module"])

            for tag in tags:
                tag_button = self.create_tag_button(f" {tag} ", "Module info", lambda checked, t=tag: self.show_tag_description(t))
                tag_container.addWidget(tag_button)

            button_container = QHBoxLayout()
            button_container.setAlignment(Qt.AlignmentFlag.AlignRight)

            if run["status"] == "completed":
                view_button = self.create_icon_button("👁", "#007bff", "Results", lambda checked, i=index: self.view_result(i))
                button_container.addWidget(view_button)

            if run["status"] == "completed":
                variation_button = self.create_icon_button("➕", "#d39e00", "New variation", lambda checked, i=index: self.create_variation(i))
                button_container.addWidget(variation_button)

            edit_button = self.create_icon_button("✎", "#d39e00", "Edit", lambda checked, i=index: self.edit_item(i))
            delete_button = self.create_icon_button("🗑", "#dc3545", "Delete", lambda checked, i=index: self.delete_item(i))

            button_container.addWidget(edit_button)
            button_container.addWidget(delete_button)

            main_row_layout = QHBoxLayout()
            main_row_layout.addWidget(description_label)
            main_row_layout.addLayout(tag_container)
            main_row_layout.addLayout(button_container)

            self.layout.addLayout(main_row_layout)

        self.content_widget.adjustSize()

    def darken_color(self, color):
        if color.startswith("#"):
            color = color[1:]
        r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        r = min(r + 30, 255)
        g = min(g + 30, 255)
        b = min(b + 30, 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    def show_tag_description(self, tag):
        """Show description of a tag."""
        msg = QMessageBox()
        msg.setWindowTitle("Module info")
        msg.setText(self.tag_descriptions.get(tag, f"No description available:<br><b>{tag}</b>"))
        msg.exec()

def format_run(run):
    return "[" + run["timestamp"] + "] " + run["description"]
