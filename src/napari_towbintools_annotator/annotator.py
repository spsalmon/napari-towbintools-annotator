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

def convert_path_to_dir_name(path):

    path = path.replace(":", "_").replace("\\", "_").replace("/", "_")
    # remove any leading underscores from the destination directory
    path = path.lstrip("_")
    path = path.lstrip("__")
    # remove any trailing underscores from the destination directory
    path = path.rstrip("_")

    return path


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

    def delete_widget(self):
        self.parent_widget.main_layout.removeWidget(self)
        self.parent_widget.project_creation_widget.hide()
        self.parent_widget.project_creation_widget = None

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

        annotations_save_dir = os.path.join(project_dir, "annotations")
        os.makedirs(annotations_save_dir, exist_ok=True)

        copy_data = self.copy_data_checkbox.isChecked()
        if copy_data:
            local_data_dir = os.path.join(project_dir, "data")
            os.makedirs(local_data_dir, exist_ok=True)

        if project_type == "classification":
            for i, data_dir in enumerate(self.data_directories):
                if os.path.isdir(data_dir):
                    d = convert_path_to_dir_name(data_dir)
                    
                    if copy_data:
                        dest_dir = os.path.join(local_data_dir, d)

                        if not os.path.exists(dest_dir):
                            shutil.copytree(data_dir, dest_dir)

            if copy_data:
                local_data_directories = [os.path.join(local_data_dir, d) for d in os.listdir(local_data_dir) if os.path.isdir(os.path.join(local_data_dir, d))]
                self.data_directories = local_data_directories

            data_files = [os.path.join(d, f) for d in self.data_directories for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
            annotation_df = pd.DataFrame({
                'ImagePath': data_files,
                'Class': [""] * len(data_files),
            })

            annotation_df_path = os.path.join(annotations_save_dir, "annotations.csv")
            annotation_df.to_csv(annotation_df_path, index=False)

            project = Project(
                name=project_name,
                image_type=image_type,
                project_type=project_type,
                annotation_directories=[annotations_save_dir],
                annotation_df_path=annotation_df_path,
                data_directories=self.data_directories,
                classes=[self.classes_list.item(i).text() for i in range(self.classes_list.count())],
                project_dir=project_dir,
            )

            project.save()

            self.delete_widget()
            self.parent_widget.load_project_from_path(project_dir)

        else:
            for i, data_dir in enumerate(self.data_directories):
                if os.path.isdir(data_dir):
                    d = convert_path_to_dir_name(data_dir)
                    annotations_dest_dir = os.path.join(annotations_save_dir, d)
                    os.makedirs(annotations_dest_dir, exist_ok=True)
                    
                    if copy_data:
                        dest_dir = os.path.join(local_data_dir, d)

                        if not os.path.exists(dest_dir):
                            shutil.copytree(data_dir, dest_dir)

            annotation_directories = [os.path.join(annotations_save_dir, convert_path_to_dir_name(d)) for d in os.listdir(annotations_save_dir) if os.path.isdir(os.path.join(annotations_save_dir, d))]
            if copy_data:
                local_data_directories = [os.path.join(local_data_dir, d) for d in os.listdir(local_data_dir) if os.path.isdir(os.path.join(local_data_dir, d))]
                self.data_directories = local_data_directories

            project = Project(
                name=project_name,
                image_type=image_type,
                project_type=project_type,
                annotation_directories=annotation_directories,
                data_directories=self.data_directories,
                classes=[self.classes_list.item(i).text() for i in range(self.classes_list.count())],
                project_dir=project_dir,
            )

            project.save()


class ClassificationAnnotatorWidget(QWidget):
    def __init__(self, napari_viewer, project, parent=None):
        super().__init__(parent=parent)

        self.viewer = napari_viewer
        self.project = project

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Create a label to display the project name
        self.project_label = QLabel(f"Project: {self.project.name}")
        self.main_layout.addWidget(self.project_label)

        self.annotation_df_path = project.annotation_df_path
        self.annotation_df = pd.read_csv(self.annotation_df_path)
        # cast class and ImagePath columns to string to avoid warnings
        self.annotation_df['Class'] = self.annotation_df['Class'].astype(str)
        self.annotation_df['ImagePath'] = self.annotation_df['ImagePath'].astype(str)

        self.data_files = sorted(self.annotation_df['ImagePath'].tolist())

        self.file_list_widget = QListWidget()
        self.file_list_widget.addItems([os.path.basename(f) for f in self.data_files])

        last_annotated_idx = -1
        if "Class" in self.annotation_df.columns and not self.annotation_df.empty:
            # Find all rows with non-empty, non-null Class values
            valid_classes = self.annotation_df["Class"].dropna().astype(str).str.strip()
            non_empty_mask = valid_classes != ""
            
            if non_empty_mask.any():
                # Get the index of the last valid annotation
                last_annotated_idx = valid_classes[non_empty_mask].index[-1] + 1

        self.current_file_idx = last_annotated_idx if last_annotated_idx != -1 else 0
        if self.current_file_idx >= len(self.data_files):
            self.current_file_idx = 0

        self.file_list_widget.setCurrentRow(self.current_file_idx)
        self._load_file()
        self.file_list_widget.itemClicked.connect(self.choose_file_from_list)

        self.main_layout.addWidget(self.file_list_widget)

        self.class_buttons_widget = QWidget()
        self.class_buttons_layout = QVBoxLayout()
        self.class_buttons_widget.setLayout(self.class_buttons_layout)

        self.class_buttons = QButtonGroup()
        for class_name in self.project.classes:
            button = QPushButton(class_name)
            self.class_buttons.addButton(button)
            self.class_buttons_layout.addWidget(button)
        self.class_buttons.buttonClicked.connect(self.assign_class)

        self.main_layout.addWidget(self.class_buttons_widget)

        self.ignore_button = QPushButton("Ignore")
        self.ignore_button.clicked.connect(self.ignore_file)
        self.main_layout.addWidget(self.ignore_button)


    def _load_file(self):
        # clear the current layers
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()

        if self.current_file_idx < 0 or self.current_file_idx >= len(self.data_files):
            print("No valid file selected.")
            return
        data_file = self.data_files[self.current_file_idx]
        class_name = self.annotation_df.loc[self.current_file_idx, 'Class']
        
        self.viewer.open(data_file, colormap="viridis", name=os.path.basename(data_file))

        if not (pd.isna(class_name) or class_name == ""):
            image_shape = np.shape(self.viewer.layers[-1].data)
            box = np.array([[0, 0], [0, image_shape[1]], [image_shape[0], image_shape[1]], [image_shape[0], 0]])

            self.viewer.add_shapes(
                box,
                shape_type='polygon',
                features = {"class": class_name},
                edge_width=3,
                text={'string': 'class', 'size': 20, 'anchor': 'upper_left', 'color': 'white'},
                face_color='transparent',
                edge_color='red',
                name=f"class name",
            )

    def choose_file_from_list(self):
        self.current_file_idx = self.file_list_widget.currentRow()
        self._load_file()

    def next_file(self):
        self.current_file_idx += 1
        if self.current_file_idx >= len(self.data_files):
            self.current_file_idx = 0
        self.file_list_widget.setCurrentRow(self.current_file_idx)
        self._load_file()

    def assign_class(self, button):
        if self.current_file_idx < 0 or self.current_file_idx >= len(self.data_files):
            print("No valid file selected.")
            return

        class_name = button.text()
        # Update the DataFrame with the assigned class
        self.annotation_df.loc[self.current_file_idx, 'Class'] = class_name

        # Save the updated DataFrame back to CSV
        self.annotation_df.to_csv(self.annotation_df_path, index=False)
        self.next_file()

    def ignore_file(self):
        if self.current_file_idx < 0 or self.current_file_idx >= len(self.data_files):
            print("No valid file selected.")
            return

        # Remove the current file from the DataFrame
        self.annotation_df.drop(index=self.current_file_idx, inplace=True)
        # Reset the index of the DataFrame
        self.annotation_df.reset_index(drop=True, inplace=True)

        # Save the updated DataFrame back to CSV
        self.annotation_df.to_csv(self.annotation_df_path, index=False)

        # Remove the file from the list and update the current index
        self.data_files.pop(self.current_file_idx)
        self.file_list_widget.takeItem(self.current_file_idx)

        if self.data_files:
            self.current_file_idx = min(self.current_file_idx, len(self.data_files) - 1)
            self.file_list_widget.setCurrentRow(self.current_file_idx)
            self._load_file()
        else:
            print("No more files to annotate.")

class TowbintoolsAnnotatorWidget(QWidget):
    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent=parent)

        self.viewer = napari_viewer

        self.main_layout = QVBoxLayout()

        self.setLayout(self.main_layout)

        self.initial_button_widget = QWidget()
        self.initial_button_layout = QVBoxLayout()
        self.initial_button_widget.setLayout(self.initial_button_layout)
        self.create_button = QPushButton("Create Project")

        self.load_button = QPushButton("Load Project")

        self.project_creation_widget = None

        self.initial_button_layout.addWidget(self.create_button)
        self.initial_button_layout.addWidget(self.load_button)
        self.main_layout.addWidget(self.initial_button_widget)

        self.create_button.clicked.connect(self.toggle_create_project)
        self.load_button.clicked.connect(self.load_project)

    def load_project(self):
        """Load an existing project"""
        project_dir = QFileDialog.getExistingDirectory(
            self,
            "Load Project Directory",
            os.getcwd(),
        )

        # Extract the directory path from the selected file
        self.load_project_from_path(project_dir)

    def load_project_from_path(self, project_dir):
        """Load a project from a given directory path"""
        if not os.path.isdir(project_dir):
            print(f"Invalid project directory: {project_dir}")
            return

        project = Project.load(project_dir)
        print(f"Loaded project: {project.name}")

        # Clear the main layout and set up the annotator widget
        self.initial_button_widget.setVisible(False)
        self.annotator_widget = create_annotator_widget(self.viewer, project, parent=self)
        self.main_layout.insertWidget(0, self.annotator_widget)



    def toggle_create_project(self):
        if self.project_creation_widget is not None:
            # Remove the widget from layout and hide it
            self.main_layout.removeWidget(self.project_creation_widget)
            self.project_creation_widget.hide()
            self.project_creation_widget = None
            
            # Show the main buttons again
            self.initial_button_widget.setVisible(True)
        else:
            # Hide the main buttons
            self.initial_button_widget.setVisible(False)
            
            # Create and insert the project creation widget at the same position
            self.project_creation_widget = ProjectCreatorWidget(self.viewer, parent=self)
            # Insert at index 0 to put it at the top, before the hidden buttons
            self.main_layout.insertWidget(0, self.project_creation_widget)
        
def create_annotator_widget(napari_viewer, project, parent=None):
    """
    Create the annotator widget based on the project type.
    """
    if project.project_type == "classification":
        return ClassificationAnnotatorWidget(napari_viewer, project, parent=parent)
    else:
        raise ValueError(f"Unsupported project type: {project.project_type}")