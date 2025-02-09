from PySide6.QtWidgets import (
    QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QComboBox,
    QFormLayout, QLineEdit, QMessageBox, QFrame, QCheckBox, QFileDialog, QDialog, QListWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator, QDoubleValidator, QIcon

def format_name(name):
    """Format parameter names for better display."""
    return name.replace("_", " ").capitalize()

class NewRun(QWidget):
    def __init__(self, stacked_widget, dataset_loaders, runs):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.dataset_loaders = dataset_loaders
        self.first_selection = True  # Track if first selection is made
        self.runs = runs

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.label = QLabel("Data", self)
        self.label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self.label)

        self.dataset_selector = QComboBox(self)
        self.dataset_selector.addItems(["Select a module"] + list(dataset_loaders.keys()))
        self.dataset_selector.currentTextChanged.connect(self.update_param_form)
        layout.addWidget(self.dataset_selector)

        # Dataset description section
        self.description_label = QLabel("Select a module to see its description and parameters to fill in.", self, openExternalLinks=True)
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("font-size: 14px; color: #555; margin-top: 5px;")
        layout.addWidget(self.description_label)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        self.param_form = QFormLayout()
        self.param_inputs = {}
        self.form_widget = QWidget()
        self.form_widget.setLayout(self.param_form)
        layout.addWidget(self.form_widget)

        button_layout = QHBoxLayout()
        self.next_button = QPushButton("Next", self)
        self.next_button.setStyleSheet("background-color: #17a2b8; color: white; padding: 6px; border-radius: 5px;")
        self.next_button.clicked.connect(self.next)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setFixedSize(80, 30)
        self.cancel_button.setStyleSheet("background-color: #dc3545; color: white; border-radius: 5px;")
        self.cancel_button.clicked.connect(self.switch_to_dashboard)

        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)
        layout.addStretch()
        self.setLayout(layout)
        self.update_param_form(self.dataset_selector.currentText())

    def showEvent(self, event):
        super().showEvent(event)
        self.dataset_selector.clear()
        self.dataset_selector.addItems(["Select a dataset loader"] + list(self.dataset_loaders.keys()))
        self.update_param_form("Select a dataset loader")

    def update_param_form(self, dataset_name):
        """Update the form based on the selected dataset loader."""
        if self.first_selection and dataset_name != "Select a dataset loader":
            self.dataset_selector.removeItem(0)
            self.first_selection = False

        for i in reversed(range(self.param_form.rowCount())):
            self.param_form.removeRow(i)
        self.param_inputs.clear()

        if dataset_name not in self.dataset_loaders:
            self.description_label.setText("Select a dataset loader to see its description.")
            return

        loader = self.dataset_loaders[dataset_name]
        self.description_label.setText(loader.get("description", "No description available."))

        # Populate parameters
        for name, param_type, default, description in loader["parameters"]:
            if name == "dataset" or name == "model": continue
            param_options = loader.get("parameter_options", {}).get(name, [])  # Get options if available
            param_widget = self.create_input_widget(name, param_type, default, description, param_options)
            self.param_form.addRow(param_widget)

    def open_sensitive_modal(self, input_field, columns):
        """Open a modal dialog to select sensitive columns."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Sensitive Columns")
        dialog.setModal(True)

        layout = QVBoxLayout()

        list_widget = QListWidget(dialog)
        list_widget.addItems(columns)
        list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        layout.addWidget(list_widget)

        confirm_button = QPushButton("Confirm", dialog)
        confirm_button.clicked.connect(lambda: self.set_sensitive_values(dialog, list_widget, input_field))
        layout.addWidget(confirm_button)

        dialog.setLayout(layout)
        dialog.exec()

    def set_sensitive_values(self, dialog, list_widget, input_field):
        """Set selected columns into the input field."""
        selected_items = [item.text() for item in list_widget.selectedItems()]
        input_field.setText(", ".join(selected_items))
        dialog.accept()

    def create_input_widget(self, name, param_type, default, description, param_options):
        """Create an appropriate input widget based on the parameter type."""
        param_layout = QHBoxLayout()

        helper = None
        """if name == "numeric" or name == "categorical":
            pass # TODO: add the
        elif name.startswith("target") or name.startswith("label"):
            col_options = self.get_cols()
            input_widget = QComboBox(self)
            input_widget.addItems(col_options)
            input_widget.setCurrentText(default if default in param_options else param_options[0])
        el"""
        if name == "sensitive":
            if not self.runs: return QWidget()
            columns = self.runs[-1]["dataset"]["return"].cols

            input_widget = QLineEdit(self)
            input_widget.setText(str(default) if default != "None" else "")

            select_button = QPushButton("...")
            select_button.setFixedSize(30, 20)
            select_button.setStyleSheet("background-color: #ddd; border-radius: 5px;")
            select_button.clicked.connect(lambda: self.open_sensitive_modal(input_widget, columns))

            helper = select_button
        elif param_options:  # If parameter options are provided, use a dropdown
            input_widget = QComboBox(self)
            input_widget.addItems(param_options)
            input_widget.setCurrentText(default if default in param_options else param_options[0])
        elif param_type == "int":
            input_widget = QLineEdit(self)
            input_widget.setValidator(QIntValidator())
            input_widget.setText(str(default) if default != "None" else "0")
        elif param_type == "float":
            input_widget = QLineEdit(self)
            input_widget.setValidator(QDoubleValidator())
            input_widget.setText(str(default) if default != "None" else "0.0")
        elif param_type == "bool":
            input_widget = QCheckBox(self)
            input_widget.setChecked(str(default).lower() == "true")
        elif param_type == "url":
            input_widget = QLineEdit(self)
            input_widget.setText(str(default) if default != "None" else "")
            file_button = QPushButton("...")
            file_button.setFixedSize(30, 20)
            file_button.setStyleSheet("background-color: #ddd; border-radius: 5px;")
            file_button.clicked.connect(lambda: self.select_path(input_widget))
            helper = file_button
        else:  # Default to a normal text field
            input_widget = QLineEdit(self)
            input_widget.setText(str(default) if default != "None" else "")

        self.param_inputs[name] = input_widget  # Store reference to input field

        label = QLabel(format_name(name))
        label.setFixedSize(150, 20)
        label.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        help_button = QPushButton("?")
        help_button.setFixedSize(30, 20)
        help_button.setStyleSheet("background-color: #ddd; border-radius: 10px; font-weight: bold;")
        help_button.clicked.connect(lambda: self.show_help_popup(format_name(name), description))

        param_layout.addWidget(label)
        param_layout.addWidget(help_button)
        if helper is not None: param_layout.addWidget(helper);
        param_layout.addWidget(input_widget)

        param_widget = QWidget()
        param_widget.setLayout(param_layout)
        return param_widget

    def select_path(self, input_field):
        """Open a file dialog to select a path."""
        path = QFileDialog.getOpenFileName(self, "Select file")
        if path: input_field.setText(path[0])

    def show_help_popup(self, param_name, description):
        """Show a popup window with the parameter description."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Help")
        msg.setText(description)
        msg.setIcon(QMessageBox.Icon.NoIcon)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        msg.exec()

    def save(self, step):
        pipeline = self.runs[-1]
        dataset_name = self.dataset_selector.currentText()
        params = {}
        for param, field in self.param_inputs.items():
            if isinstance(field, QCheckBox):
                params[param] = field.isChecked()
            elif isinstance(field, QComboBox):
                params[param] = field.currentText()
            else:
                params[param] = field.text()
        pipeline[step] = {"module": dataset_name, "params": params}

    def show_error_message(self, message):
        error_msg = QMessageBox(self)
        error_msg.setWindowTitle("Error")
        error_msg.setText(message)
        error_msg.setIcon(QMessageBox.Critical)
        error_msg.setModal(True)
        error_msg.exec()