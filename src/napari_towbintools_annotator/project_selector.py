import os
from pathlib import Path

import napari
import numpy as np
import pandas as pd
from magicgui.widgets import create_widget
from napari.layers.points._points_constants import Mode
from napari_guitils.gui_structures import TabSet, VHGroup
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QLabel,
    QLineEdit,
    QTextEdit,
    QCheckBox,
    QListWidget,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)
from .project import Project
import datetime
import shutil

class ProjectCreatorWidget(QWidget):
    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent=parent)

        self.viewer = napari_viewer
        self.parent_widget = parent

        self.data_directories = []
        self.project_dir = []

        # creates the elements that will be used to create a new project, and are hidden until the button is clicked
        self.project_creation_layout = QVBoxLayout()
        self.setLayout(self.project_creation_layout)

        # Project name input
        self.project_name_input = QLineEdit()
        self.project_name_input.setText(f'{datetime.datetime.now().strftime("%Y%m%d")}_project')
        self.project_name_input.setClearButtonEnabled(False)
        self.project_creation_layout.addWidget(self.project_name_input)

        # image type selector
        image_type_group = VHGroup("Image Type", orientation="G")
        self.image_type_selector = QButtonGroup()
        self.image_type_multichannel = QRadioButton("Multichannel")
        self.image_type_multichannel.setChecked(True)
        self.image_type_zstack = QRadioButton("Z-Stack")
        self.image_type_time_series = QRadioButton("Time Series")

        self.image_type_selector.addButton(self.image_type_multichannel)
        self.image_type_selector.addButton(self.image_type_zstack)
        self.image_type_selector.addButton(self.image_type_time_series)

        self.image_type_multichannel.setToolTip("Multichannel images (e.g., RGB, RGBA)")
        self.image_type_zstack.setToolTip("Z-Stack images (e.g., multiple z-slices of the same image)")
        self.image_type_time_series.setToolTip("Time Series images (e.g., multiple time points of the same image)")

        image_type_group.glayout.addWidget(self.image_type_multichannel)
        image_type_group.glayout.addWidget(self.image_type_zstack)
        image_type_group.glayout.addWidget(self.image_type_time_series)

        self.project_creation_layout.addWidget(image_type_group.gbox)

        # project type selector
        self.project_type_layout = VHGroup("Project Type", orientation="G")
        self.project_type_selector = QButtonGroup()
        self.project_type_classification = QRadioButton("Classification")
        self.project_type_classification.setChecked(True)
        self.project_type_keypoint = QRadioButton("Keypoint")
        self.project_type_panoptic = QRadioButton("Panoptic")

        self.project_type_selector.addButton(self.project_type_classification)
        self.project_type_selector.addButton(self.project_type_keypoint)
        self.project_type_selector.addButton(self.project_type_panoptic)

        self.project_type_classification.setToolTip("Classification project (e.g., image classification)")
        self.project_type_keypoint.setToolTip("Keypoint project (e.g., keypoint detection)")
        self.project_type_panoptic.setToolTip("Panoptic project (e.g., panoptic segmentation)")

        self.project_type_layout.glayout.addWidget(self.project_type_classification)
        self.project_type_layout.glayout.addWidget(self.project_type_keypoint)
        self.project_type_layout.glayout.addWidget(self.project_type_panoptic)

        self.project_creation_layout.addWidget(self.project_type_layout.gbox)

        # dir selection
        self.dir_selection_layout = VHGroup("Directory Selection", orientation="G")

        def create_dir_selector(parent, button_label, dir_list=None, multiple=False):
            dir_selector_widget = QWidget()
            dir_selector_layout = QVBoxLayout()
            buttons_layout = QHBoxLayout()
            dir_button = QPushButton(button_label)
        
            if multiple:
                dir_display = QListWidget()
                dir_display.setSelectionMode(QListWidget.MultiSelection)
                remove_button = QPushButton("Remove Selected")
                remove_button.clicked.connect(lambda: remove_selected_directories(dir_display, dir_list))
                
                buttons_layout.addWidget(dir_button)
                buttons_layout.addWidget(remove_button)
                dir_selector_layout.addLayout(buttons_layout)
                dir_selector_layout.addWidget(dir_display)
                
            else:
                dir_display = QLabel()

                dir_selector_layout.addWidget(dir_button)
                dir_selector_layout.addLayout(buttons_layout)
                dir_selector_layout.addWidget(dir_display)

            dir_button.clicked.connect(
                lambda: add_directory_to_list(
                    QFileDialog.getExistingDirectory(
                        parent, "Select Directory", os.getcwd()
                    ),
                    dir_display,
                    dir_list,
                    multiple=multiple
                )
            )
            
            dir_selector_widget.setLayout(dir_selector_layout)
            return dir_selector_widget
        
        def add_directory_to_list(dir_path, display_widget, data_list=None, multiple=False):
            if dir_path and os.path.isdir(dir_path):
                if isinstance(data_list, list):
                    if multiple:
                        # Append to the list if multiple directories are allowed
                        data_list.append(dir_path)
                    else:
                        # Replace the single directory path
                        data_list.clear()
                        data_list.append(dir_path)

                if isinstance(display_widget, QListWidget):
                    display_widget.addItem(dir_path)
                elif isinstance(display_widget, QLabel):
                    display_widget.setText(dir_path)
                else:
                    print("Unsupported display widget type")
            else:
                print(f"Invalid directory path: {dir_path}")

        def remove_selected_directories(display_widget, data_list):
            if isinstance(display_widget, QListWidget):
                for item in display_widget.selectedItems():
                    display_widget.takeItem(display_widget.row(item))
                    data_list.remove(item.text())

        # Wrap data_selection_layout in a QWidget before adding to another layout
        self.data_selection_widget = create_dir_selector(
            self,
            "Select Data Directory",
            dir_list=self.data_directories,
            multiple=True
        )
        self.dir_selection_layout.glayout.addWidget(self.data_selection_widget)

        # Annotations directory selection
        self.project_dir_selection_layout = create_dir_selector(
            self,
            "Select Project Directory",
            dir_list=self.project_dir,
            multiple=False
        )
        self.dir_selection_layout.glayout.addWidget(self.project_dir_selection_layout)

        self.project_creation_layout.addWidget(self.dir_selection_layout.gbox)

        # specifying the options for classification project
        self.classification_options_layout = VHGroup("Classification", orientation="G")
        self.classes_layout = VHGroup("Classes", orientation="G")
        # Widget to display and manage classes
        self.classes_list = QListWidget()
        self.classes_list.setSelectionMode(QListWidget.MultiSelection)

        # Input for adding new class
        self.class_input_layout = QHBoxLayout()
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Add new class...")
        self.add_class_button = QPushButton("Add")
        self.remove_class_button = QPushButton("Remove Selected")

        self.class_input_layout.addWidget(self.class_input)
        self.class_input_layout.addWidget(self.add_class_button)
        self.class_input_layout.addWidget(self.remove_class_button)

        # Add input layout above the list
        self.classes_layout.glayout.addLayout(self.class_input_layout, 0, 0)

        # Add class on button click or Enter
        def add_class():
            class_name = self.class_input.text().strip()
            if class_name and not any(self.classes_list.item(i).text() == class_name for i in range(self.classes_list.count())):
                self.classes_list.addItem(class_name)
                self.class_input.clear()

        self.add_class_button.clicked.connect(add_class)
        self.class_input.returnPressed.connect(add_class)

        # Remove selected classes
        def remove_selected_classes():
            for item in self.classes_list.selectedItems():
                self.classes_list.takeItem(self.classes_list.row(item))

        self.remove_class_button.clicked.connect(remove_selected_classes)
        self.classes_list.setToolTip("Select classes for the classification project")
        self.classes_layout.glayout.addWidget(self.classes_list)
        self.classification_options_layout.glayout.addWidget(self.classes_layout.gbox)
        self.project_creation_layout.addWidget(self.classification_options_layout.gbox)

        # Show the classification options by default
        self.classification_options_layout.gbox.setVisible(True)
        self.project_type_selector.buttonClicked.connect(self.toggle_project_type_options)

        # create and cancel buttons
        self.copy_data_checkbox = QCheckBox("Copy data to project directory")
        self.create_button = QPushButton("Create Project")
        self.cancel_button = QPushButton("Cancel")

        self.create_button.clicked.connect(self.create_project)
        self.cancel_button.clicked.connect(self.cancel_creation)

        self.project_creation_layout.addWidget(self.copy_data_checkbox)
        self.project_creation_layout.addWidget(self.create_button)
        self.project_creation_layout.addWidget(self.cancel_button)


    def toggle_project_type_options(self):
        if self.project_type_classification.isChecked():
            # Show classification options
            self.classification_options_layout.gbox.setVisible(True)
        else:
            # Hide classification options
            self.classification_options_layout.gbox.setVisible(False)

    def cancel_creation(self):
        """Cancel project creation and return to main view"""
        if self.parent_widget and hasattr(self.parent_widget, 'toggle_create_project'):
            # Use Qt's queued connection to avoid immediate widget destruction issues
            QTimer.singleShot(0, self.parent_widget.toggle_create_project)

    def create_project(self):
        """Create the project with the selected options"""
        project_name = self.project_name_input.text().strip()
        
        if not project_name:
            print("Please enter a project name")
            return
            
        # Get selected image type
        image_type = "multichannel"
        if self.image_type_zstack.isChecked():
            image_type = "zstack"
        elif self.image_type_time_series.isChecked():
            image_type = "time_series"
            
        # Get selected project type
        project_type = "classification"
        if self.project_type_keypoint.isChecked():
            project_type = "keypoint"
        elif self.project_type_panoptic.isChecked():
            project_type = "panoptic"

        project_dir = os.path.join(
            self.project_dir[0] if self.project_dir else "",
            f"{project_name}"
        )

        os.makedirs(project_dir, exist_ok=True)

        images_to_annotate = []
        if self.copy_data_checkbox.isChecked():
            local_data_dir = os.path.join(project_dir, "data")
            os.makedirs(local_data_dir, exist_ok=True)
            for i, data_dir in enumerate(self.data_directories):
                if os.path.isdir(data_dir):
                    dest_dir = os.path.join(
                        local_data_dir,
                        data_dir.replace(":", "_").replace("\\", "_").replace("/", "_")
                    )
                    # remove any leading underscores from the destination directory
                    dest_dir = dest_dir.lstrip("_")
                    # remove any trailing underscores from the destination directory
                    dest_dir = dest_dir.rstrip("_")

                    dest_dir = os.path.join(local_data_dir, dest_dir)

                    if not os.path.exists(dest_dir):
                        shutil.copytree(data_dir, dest_dir)

            local_data_directories = [os.path.join(local_data_dir, d) for d in os.listdir(local_data_dir) if os.path.isdir(os.path.join(local_data_dir, d))]
            self.data_directories = local_data_directories


        for data_dir in self.data_directories:
            if os.path.isdir(data_dir):
                images_to_annotate.extend(
                    [os.path.join(data_dir, img) for img in os.listdir(data_dir)]
                )

        annotations_save_dir = os.path.join(project_dir, "annotations")
        os.makedirs(annotations_save_dir, exist_ok=True)

        project = Project(
            name=project_name,
            image_type=image_type,
            project_type=project_type,
            data_directories=self.data_directories,
            project_dir=project_dir,
            images_to_annotate=images_to_annotate,
        )

        print(project)

class ProjectSelectorWidget(QWidget):
    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent=parent)

        self.viewer = napari_viewer

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.create_button = QPushButton("Create Project")

        self.load_button = QPushButton("Load Project")

        self.project_creation_widget = None

        self.main_layout.addWidget(self.create_button)
        self.main_layout.addWidget(self.load_button)

        self.create_button.clicked.connect(self.toggle_create_project)

    def toggle_create_project(self):
        if self.project_creation_widget is not None:
            # Remove the widget from layout and hide it
            self.main_layout.removeWidget(self.project_creation_widget)
            self.project_creation_widget.hide()
            self.project_creation_widget = None
            
            # Show the main buttons again
            self.create_button.setVisible(True)
            self.load_button.setVisible(True)
        else:
            # Hide the main buttons
            self.create_button.setVisible(False)
            self.load_button.setVisible(False)
            
            # Create and insert the project creation widget at the same position
            self.project_creation_widget = ProjectCreatorWidget(self.viewer, parent=self)
            # Insert at index 0 to put it at the top, before the hidden buttons
            self.main_layout.insertWidget(0, self.project_creation_widget)
        