import os
from pathlib import Path

import napari
import numpy as np
import pandas as pd
from magicgui.widgets import create_widget
from napari.layers.points._points_constants import Mode
from napari_guitils.gui_structures import TabSet, VHGroup
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QLabel,
    QLineEdit,
    QTextEdit,
    QListWidget,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

from napari_guitils.gui_structures import TabSet, VHGroup
import datetime

class ProjectCreatorWidget(QWidget):
    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent=parent)

        self.viewer = napari_viewer
        self.parent_widget = parent

        self.data_directories = []
        self.annotations_directory = None

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

        def create_dir_selector(parent, layout, button_label, connect_function=None):
            dir_button = QPushButton(button_label)
            dir_button.clicked.connect(
                lambda: connect_function(parent)
            )
            return dir_button
        
        def select_data_directory(parent):
            dir_path = QFileDialog.getExistingDirectory(
                parent, "Select Directory"
            )
            if dir_path:
                parent.data_directories.append(dir_path)

                # add a line edit to display the selected directory and a button to clear it
                dir_label = QLabel(dir_path)
                clear_button = QPushButton("Clear")
                
                # Create a horizontal layout to put label and clear button on the same line
                dir_row_widget = QWidget()
                dir_row_layout = QHBoxLayout(dir_row_widget)
                dir_row_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for a cleaner look
                dir_row_layout.addWidget(dir_label)
                dir_row_layout.addWidget(clear_button)
                
                def clear_dir():
                    if dir_path in parent.data_directories:
                        parent.data_directories.remove(dir_path)
                    parent.data_selection_layout.removeWidget(dir_row_widget)
                    dir_row_widget.setParent(None)
                    dir_row_widget.deleteLater()
                clear_button.clicked.connect(clear_dir)
                
                parent.data_selection_layout.addWidget(dir_row_widget)
                    
                return dir_path
            else:
                print("Directory selection cancelled.")
                return ""
            
        def select_annotations_directory(parent):
            dir_path = QFileDialog.getExistingDirectory(
                parent, "Select Annotations Directory"
            )
            if dir_path:
                self.annotations_directory = dir_path

                if not hasattr(parent, 'annotations_dir_label'):
                    parent.annotations_dir_label = QLabel(dir_path)
                    parent.dir_selection_layout.glayout.addWidget(parent.annotations_dir_label)
                else:
                    parent.annotations_dir_label.setText(dir_path)
                    
                return dir_path
            else:
                print("Directory selection cancelled.")
                return ""
            
        # Wrap data_selection_layout in a QWidget before adding to another layout
        self.data_selection_widget = QWidget()
        self.data_selection_layout = QVBoxLayout(self.data_selection_widget)
        self.data_dir_button = create_dir_selector(
            self, self.dir_selection_layout, "Select Data Directory", select_data_directory
        )
        self.data_selection_layout.addWidget(self.data_dir_button)

        self.annotations_dir_button = create_dir_selector(
            self, self.dir_selection_layout, "Select Annotations Directory", select_annotations_directory
        )

        self.dir_selection_layout.glayout.addWidget(self.data_selection_widget)
        self.dir_selection_layout.glayout.addWidget(self.annotations_dir_button)

        self.project_creation_layout.addWidget(self.dir_selection_layout.gbox)

        # create and cancel buttons
        self.create_button = QPushButton("Create Project")
        self.cancel_button = QPushButton("Cancel")

        self.create_button.clicked.connect(self.create_project)
        self.cancel_button.clicked.connect(self.cancel_creation)

        self.project_creation_layout.addWidget(self.create_button)
        self.project_creation_layout.addWidget(self.cancel_button)

    def cancel_creation(self):
        """Cancel project creation and return to main view"""
        if self.parent_widget and hasattr(self.parent_widget, 'toggle_create_project'):
            # Use Qt's queued connection to avoid immediate widget destruction issues
            from qtpy.QtCore import QTimer
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
            
        print(f"Creating project: {project_name}, Image type: {image_type}, Project type: {project_type}")
        print(f"Data directories: {self.data_directories}")
        print(f"Annotations directory: {self.annotations_directory}")

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
        