from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
    QSizePolicy, QSpacerItem, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWebEngineWidgets import QWebEngineView


def format_run(run):
    return "[" + run["timestamp"] + "] " + run["description"]


class Results(QWidget):
    def __init__(self, stacked_widget, runs, tag_descriptions):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.runs = runs
        self.tag_descriptions = tag_descriptions

        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Title label
        self.title_label = QLabel("Analysis Outcome", self)
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.layout.addWidget(self.title_label)

        # Container for buttons and tags
        self.top_container = QHBoxLayout()
        self.top_container.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.top_container.setSpacing(4)  # Reduced spacing between buttons

        # Buttons (Square Icons with Tooltips)
        self.edit_button = self.create_icon_button("✎", "#17a2b8", "Edit this analysis", self.edit_run)
        self.variation_button = self.create_icon_button("➕", "#ffc107", "Create a variation", self.create_variation)
        self.delete_button = self.create_icon_button("🗑", "#dc3545", "Delete this result", self.delete_run)
        self.close_button = self.create_icon_button("❌", "#6c757d", "Close and return to dashboard", self.switch_to_dashboard)

        self.top_container.addWidget(self.edit_button)
        self.top_container.addWidget(self.variation_button)
        self.top_container.addWidget(self.delete_button)
        self.top_container.addWidget(self.close_button)

        # Spacer to separate buttons and tags
        self.top_container.addItem(QSpacerItem(10, 10, QSizePolicy.Fixed, QSizePolicy.Minimum))

        # Tags container (Now aligned to the left)
        self.tags_container = QHBoxLayout()
        self.tags_container.setAlignment(Qt.AlignmentFlag.AlignLeft)  # Left-aligned

        self.top_container.addLayout(self.tags_container)

        self.layout.addLayout(self.top_container)  # Add buttons and tags container

        # Results Viewer
        self.results_viewer = QWebEngineView(self)
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.results_viewer.setSizePolicy(size_policy)
        self.layout.addWidget(self.results_viewer, 1)  # "1" ensures it stretches to available space

        self.setLayout(self.layout)

    def create_icon_button(self, text, color, tooltip, callback):
        """Helper function to create square buttons with icons and tooltips."""
        button = QPushButton(text, self)
        button.setStyleSheet(f"background-color: {color}; color: white; border-radius: 5px;")
        button.setFixedSize(30, 30)
        button.setToolTip(tooltip)  # Set tooltip (hint)
        button.clicked.connect(callback)
        return button

    def switch_to_dashboard(self):
        self.stacked_widget.setCurrentIndex(0)

    def showEvent(self, event):
        super().showEvent(event)

        # Update title and results
        if self.runs:
            run = self.runs[-1]
            self.title_label.setText(format_run(run))
            html_content = run["analysis"].get("return", "<p>No results available.</p>")
            self.update_tags(run)  # Update tags
        else:
            html_content = "<p>No results available.</p>"

        # Use QTimer to ensure WebEngineView renders properly
        QTimer.singleShot(100, lambda: self.results_viewer.setHtml(html_content))
        self.results_viewer.show()

    def update_tags(self, run):
        """Refresh the tags displayed below the title."""
        # Clear existing tags
        while self.tags_container.count():
            item = self.tags_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Get tags
        tags = []
        if "dataset" in run: tags.append(run["dataset"]["module"])
        if "model" in run: tags.append(run["model"]["module"])
        if "analysis" in run: tags.append(run["analysis"]["module"])

        for tag in tags:
            tag_button = QPushButton(f" {tag} ", self)
            tag_button.setStyleSheet("background-color: gray; color: white; padding: 2px 6px; border-radius: 10px;")
            tag_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            tag_button.setToolTip("Click to view tag description")  # Tooltip hint
            tag_button.clicked.connect(lambda checked, t=tag: self.show_tag_description(t))
            self.tags_container.addWidget(tag_button)

    def show_tag_description(self, tag):
        """Show description of a tag."""
        msg = QMessageBox()
        msg.setWindowTitle("Help")
        msg.setText(self.tag_descriptions.get(tag, "No description available."))
        msg.exec()

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
