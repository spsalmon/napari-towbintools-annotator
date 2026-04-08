import datetime
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from napari_guitils.gui_structures import VHGroup
from natsort import natsorted
from qtpy.QtCore import QThread
from qtpy.QtCore import QTimer
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QButtonGroup
from qtpy.QtWidgets import QCheckBox
from qtpy.QtWidgets import QFileDialog
from qtpy.QtWidgets import QHBoxLayout
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QLineEdit
from qtpy.QtWidgets import QListWidget
from qtpy.QtWidgets import QMessageBox
from qtpy.QtWidgets import QPushButton
from qtpy.QtWidgets import QRadioButton
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget

from .annotators import ClassificationAnnotatorWidget
from .project import ClassificationProject
from .project import Project


def convert_path_to_dir_name(path):
    path = path.replace(":", "_").replace("\\", "_").replace("/", "_")
    path = path.lstrip("_")
    path = path.rstrip("_")
    return path


class ProjectCreationWorker(QThread):
    status = Signal(str)
    finished = Signal(str)  # emits project_dir on success
    error = Signal(str)

    def __init__(self, task, parent=None):
        super().__init__(parent=parent)
        self._task = task

    def run(self):
        try:
            project_dir = self._task(self.status)
            self.finished.emit(project_dir)
        except Exception as e:
            self.error.emit(str(e))


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
        project_dir = QFileDialog.getExistingDirectory(
            self, "Load Project Directory", os.getcwd()
        )
        self.load_project_from_path(project_dir)

    def load_project_from_path(self, project_dir):
        if not os.path.isdir(project_dir):
            print(f"Invalid project directory: {project_dir}")
            return

        project = Project.load(project_dir)
        self.initial_button_widget.setVisible(False)
        self.annotator_widget = create_annotator_widget(
            self.viewer, project, parent=self
        )
        self.main_layout.insertWidget(0, self.annotator_widget)

    def toggle_create_project(self):
        if self.project_creation_widget is not None:
            self.main_layout.removeWidget(self.project_creation_widget)
            self.project_creation_widget.hide()
            self.project_creation_widget = None
            self.initial_button_widget.setVisible(True)
        else:
            self.initial_button_widget.setVisible(False)
            self.project_creation_widget = ProjectCreatorWidget(
                self.viewer, parent=self
            )
            self.main_layout.insertWidget(0, self.project_creation_widget)


def create_annotator_widget(napari_viewer, project, parent=None):
    if project.project_type == "classification":
        return ClassificationAnnotatorWidget(
            napari_viewer, project, parent=parent
        )
    raise NotImplementedError(
        f"Unsupported project type: {project.project_type}"
    )


class ProjectCreatorWidget(QWidget):
    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent=parent)

        self.viewer = napari_viewer
        self.parent_widget = parent

        self.data_directories = []
        self.project_dir = []

        self.project_creation_layout = QVBoxLayout()
        self.setLayout(self.project_creation_layout)

        self.project_name_input = QLineEdit()
        self.project_name_input.setText(
            f'{datetime.datetime.now().strftime("%Y%m%d")}_project'
        )
        self.project_name_input.setClearButtonEnabled(False)
        self.project_creation_layout.addWidget(self.project_name_input)

        image_type_group = VHGroup("Image Type", orientation="G")
        self.image_type_selector = QButtonGroup()
        self.image_type_multichannel = QRadioButton("Multichannel")
        self.image_type_multichannel.setChecked(True)
        self.image_type_zstack = QRadioButton("Z-Stack")
        self.image_type_time_series = QRadioButton("Time Series")

        self.image_type_selector.addButton(self.image_type_multichannel)
        self.image_type_selector.addButton(self.image_type_zstack)
        self.image_type_selector.addButton(self.image_type_time_series)

        self.image_type_multichannel.setToolTip(
            "Multichannel images (e.g., RGB, RGBA)"
        )
        self.image_type_zstack.setToolTip(
            "Z-Stack images (e.g., multiple z-slices of the same image)"
        )
        self.image_type_time_series.setToolTip(
            "Time Series images (e.g., multiple time points of the same image)"
        )

        image_type_group.glayout.addWidget(self.image_type_multichannel)
        image_type_group.glayout.addWidget(self.image_type_zstack)
        image_type_group.glayout.addWidget(self.image_type_time_series)

        self.project_creation_layout.addWidget(image_type_group.gbox)

        self.project_type_layout = VHGroup("Project Type", orientation="G")
        self.project_type_selector = QButtonGroup()
        self.project_type_classification = QRadioButton("Classification")
        self.project_type_classification.setChecked(True)
        self.project_type_keypoint = QRadioButton("Keypoint")
        self.project_type_panoptic = QRadioButton("Panoptic")

        self.project_type_selector.addButton(self.project_type_classification)
        self.project_type_selector.addButton(self.project_type_keypoint)
        self.project_type_selector.addButton(self.project_type_panoptic)

        self.project_type_classification.setToolTip(
            "Classification project (e.g., image classification)"
        )
        self.project_type_keypoint.setToolTip(
            "Keypoint project (e.g., keypoint detection)"
        )
        self.project_type_panoptic.setToolTip(
            "Panoptic project (e.g., panoptic segmentation)"
        )

        self.project_type_layout.glayout.addWidget(
            self.project_type_classification
        )
        self.project_type_layout.glayout.addWidget(self.project_type_keypoint)
        self.project_type_layout.glayout.addWidget(self.project_type_panoptic)

        # Display mode — classification only, lives under project type selector
        self.mask_directories = []
        self.display_mode_group = VHGroup("Display Mode", orientation="G")
        self.display_mode_selector = QButtonGroup()
        self.display_mode_image = QRadioButton("Image only")
        self.display_mode_image.setChecked(True)
        self.display_mode_mask = QRadioButton("Mask only")
        self.display_mode_both = QRadioButton("Both")

        self.display_mode_selector.addButton(self.display_mode_image)
        self.display_mode_selector.addButton(self.display_mode_mask)
        self.display_mode_selector.addButton(self.display_mode_both)

        self.display_mode_group.glayout.addWidget(self.display_mode_image)
        self.display_mode_group.glayout.addWidget(self.display_mode_mask)
        self.display_mode_group.glayout.addWidget(self.display_mode_both)

        self.project_type_layout.glayout.addWidget(
            self.display_mode_group.gbox
        )
        self.display_mode_group.gbox.setVisible(True)

        self.project_type_selector.buttonClicked.connect(
            self.toggle_project_type_options
        )
        self.display_mode_selector.buttonClicked.connect(
            self._toggle_mask_dir_selector
        )

        self.project_creation_layout.addWidget(self.project_type_layout.gbox)

        self.dir_selection_layout = VHGroup(
            "Directory Selection", orientation="G"
        )

        self.data_selection_widget = self._create_dir_selector(
            self,
            "Select Image Directory",
            dir_list=self.data_directories,
            multiple=True,
        )
        self.dir_selection_layout.glayout.addWidget(self.data_selection_widget)

        # Mask directory selector — shown only when mask or both is selected
        self.mask_dir_selector_widget = self._create_dir_selector(
            self,
            "Select Mask Directory",
            dir_list=self.mask_directories,
            multiple=True,
        )
        self.mask_dir_selector_widget.setVisible(False)
        self.dir_selection_layout.glayout.addWidget(
            self.mask_dir_selector_widget
        )

        self.project_dir_selection_layout = self._create_dir_selector(
            self,
            "Select Storage Directory",
            dir_list=self.project_dir,
            multiple=False,
        )
        self.dir_selection_layout.glayout.addWidget(
            self.project_dir_selection_layout
        )

        self.project_creation_layout.addWidget(self.dir_selection_layout.gbox)

        # --- Classification-specific options ---
        self.classification_options_layout = VHGroup(
            "Classification", orientation="G"
        )

        self.classes_layout = VHGroup("Classes", orientation="G")
        self.classes_list = QListWidget()
        self.classes_list.setSelectionMode(QListWidget.MultiSelection)

        self.class_input_layout = QHBoxLayout()
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Add new class...")
        self.add_class_button = QPushButton("Add")
        self.remove_class_button = QPushButton("Remove Selected")

        self.class_input_layout.addWidget(self.class_input)
        self.class_input_layout.addWidget(self.add_class_button)
        self.class_input_layout.addWidget(self.remove_class_button)

        self.add_class_button.clicked.connect(self._add_class)
        self.class_input.returnPressed.connect(self._add_class)
        self.remove_class_button.clicked.connect(self._remove_selected_classes)

        self.classes_layout.glayout.addLayout(self.class_input_layout, 0, 0)
        self.classes_layout.glayout.addWidget(self.classes_list)
        self.classification_options_layout.glayout.addWidget(
            self.classes_layout.gbox
        )

        self.project_creation_layout.addWidget(
            self.classification_options_layout.gbox
        )
        self.classification_options_layout.gbox.setVisible(True)

        self.copy_data_checkbox = QCheckBox("Copy data to project directory")
        self.create_button = QPushButton("Create Project")
        self.cancel_button = QPushButton("Cancel")
        self.status_label = QLabel("")

        self.create_button.clicked.connect(self.create_project)
        self.cancel_button.clicked.connect(self.cancel_creation)

        self.project_creation_layout.addWidget(self.copy_data_checkbox)
        self.project_creation_layout.addWidget(self.create_button)
        self.project_creation_layout.addWidget(self.cancel_button)
        self.project_creation_layout.addWidget(self.status_label)

    def toggle_project_type_options(self):
        is_classification = self.project_type_classification.isChecked()
        self.classification_options_layout.gbox.setVisible(is_classification)
        self.display_mode_group.gbox.setVisible(is_classification)
        if not is_classification:
            self.mask_dir_selector_widget.setVisible(False)
            self.data_selection_widget.setVisible(True)

    def _toggle_mask_dir_selector(self):
        needs_masks = (
            self.display_mode_mask.isChecked()
            or self.display_mode_both.isChecked()
        )
        self.mask_dir_selector_widget.setVisible(needs_masks)
        self.data_selection_widget.setVisible(
            not self.display_mode_mask.isChecked()
        )

    def _add_class(self):
        class_name = self.class_input.text().strip()
        existing = [
            self.classes_list.item(i).text()
            for i in range(self.classes_list.count())
        ]
        if class_name and class_name not in existing:
            self.classes_list.addItem(class_name)
            self.class_input.clear()

    def _remove_selected_classes(self):
        for item in self.classes_list.selectedItems():
            self.classes_list.takeItem(self.classes_list.row(item))

    @staticmethod
    def _add_directory_to_list(
        dir_path, display_widget, data_list=None, multiple=False
    ):
        if not (dir_path and os.path.isdir(dir_path)):
            return
        if isinstance(data_list, list):
            if multiple:
                data_list.append(dir_path)
            else:
                data_list.clear()
                data_list.append(dir_path)
        if isinstance(display_widget, QListWidget):
            display_widget.addItem(dir_path)
        elif isinstance(display_widget, QLabel):
            display_widget.setText(dir_path)

    @staticmethod
    def _remove_selected_directories(display_widget, data_list):
        for item in display_widget.selectedItems():
            display_widget.takeItem(display_widget.row(item))
            data_list.remove(item.text())

    @staticmethod
    def _create_dir_selector(
        parent, button_label, dir_list=None, multiple=False
    ):
        dir_selector_widget = QWidget()
        dir_selector_layout = QVBoxLayout()
        buttons_layout = QHBoxLayout()
        dir_button = QPushButton(button_label)

        if multiple:
            dir_display = QListWidget()
            dir_display.setSelectionMode(QListWidget.MultiSelection)
            remove_button = QPushButton("Remove Selected")
            remove_button.clicked.connect(
                lambda: ProjectCreatorWidget._remove_selected_directories(
                    dir_display, dir_list
                )
            )
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
            lambda: ProjectCreatorWidget._add_directory_to_list(
                QFileDialog.getExistingDirectory(
                    parent, "Select Directory", os.getcwd()
                ),
                dir_display,
                dir_list,
                multiple=multiple,
            )
        )

        dir_selector_widget.setLayout(dir_selector_layout)
        return dir_selector_widget

    def cancel_creation(self):
        if self.parent_widget and hasattr(
            self.parent_widget, "toggle_create_project"
        ):
            QTimer.singleShot(0, self.parent_widget.toggle_create_project)

    def delete_widget(self):
        self.parent_widget.main_layout.removeWidget(self)
        self.parent_widget.project_creation_widget.hide()
        self.parent_widget.project_creation_widget = None

    def _get_selected_image_type(self):
        if self.image_type_zstack.isChecked():
            return "zstack"
        if self.image_type_time_series.isChecked():
            return "time_series"
        return "multichannel"

    def _get_selected_project_type(self):
        if self.project_type_keypoint.isChecked():
            return "keypoint"
        if self.project_type_panoptic.isChecked():
            return "panoptic"
        return "classification"

    def _get_classes(self):
        return [
            self.classes_list.item(i).text()
            for i in range(self.classes_list.count())
        ]

    def _get_display_mode(self):
        if self.display_mode_mask.isChecked():
            return "mask"
        if self.display_mode_both.isChecked():
            return "both"
        return "image"

    def _show_error(self, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle("Validation Error")
        msg.exec_()

    def create_project(self):
        project_name = self.project_name_input.text().strip()
        if not project_name:
            self._show_error("Please enter a project name.")
            return

        image_type = self._get_selected_image_type()
        project_type = self._get_selected_project_type()
        copy_data = self.copy_data_checkbox.isChecked()

        project_dir = os.path.join(
            self.project_dir[0] if self.project_dir else Path.home(),
            project_name,
        )

        if project_type == "classification":
            display_mode = self._get_display_mode()
            if display_mode in ("mask", "both") and not self.mask_directories:
                self._show_error("No mask directories selected.")
                return

            # Capture state needed by the worker before it runs
            data_directories = list(self.data_directories)
            mask_directories = list(self.mask_directories)
            classes = self._get_classes()

            def task(status):
                return self._run_classification_creation(
                    project_name,
                    image_type,
                    display_mode,
                    project_dir,
                    data_directories,
                    mask_directories,
                    classes,
                    copy_data,
                    status,
                )

        else:
            data_directories = list(self.data_directories)

            def task(status):
                return self._run_other_creation(
                    project_name,
                    image_type,
                    project_type,
                    project_dir,
                    data_directories,
                    copy_data,
                    status,
                )

        self.create_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Creating project...")

        self._worker = ProjectCreationWorker(task, parent=self)
        self._worker.status.connect(self.status_label.setText)
        self._worker.finished.connect(self._on_creation_finished)
        self._worker.error.connect(self._on_creation_error)
        self._worker.start()

    def _on_creation_finished(self, project_dir):
        self.status_label.setText("")
        self.create_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.delete_widget()
        self.parent_widget.load_project_from_path(project_dir)

    def _on_creation_error(self, message):
        self.status_label.setText("")
        self.create_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self._show_error(message)

    def _run_classification_creation(
        self,
        project_name,
        image_type,
        display_mode,
        project_dir,
        data_directories,
        mask_directories,
        classes,
        copy_data,
        status,
    ):
        os.makedirs(project_dir, exist_ok=True)
        annotations_save_dir = os.path.join(project_dir, "annotations")
        os.makedirs(annotations_save_dir, exist_ok=True)

        if copy_data:
            local_data_dir = os.path.join(project_dir, "data")
            os.makedirs(local_data_dir, exist_ok=True)
            status.emit("Copying data...")
            data_directories = self._copy_data_directories_static(
                data_directories, local_data_dir, status
            )

        mask_files = []
        if display_mode in ("mask", "both") and mask_directories:
            status.emit("Scanning mask files...")
            mask_files = natsorted(
                [
                    os.path.join(d, f)
                    for d in mask_directories
                    for f in os.listdir(d)
                    if os.path.isfile(os.path.join(d, f))
                ]
            )

        if display_mode == "mask":
            annotation_df_data = {
                "MaskPath": mask_files,
                "Class": [np.nan] * len(mask_files),
            }
        else:
            status.emit("Scanning image files...")
            data_files = natsorted(
                [
                    os.path.join(d, f)
                    for d in data_directories
                    for f in os.listdir(d)
                    if os.path.isfile(os.path.join(d, f))
                ]
            )
            if display_mode == "both" and len(data_files) != len(mask_files):
                raise ValueError(
                    f"File count mismatch: {len(data_files)} image files vs "
                    f"{len(mask_files)} mask files."
                )
            annotation_df_data = {
                "ImagePath": data_files,
                "Class": [np.nan] * len(data_files),
            }
            if mask_files:
                annotation_df_data["MaskPath"] = mask_files

        status.emit("Writing annotation file...")
        annotation_df = pd.DataFrame(annotation_df_data)
        annotation_df_path = os.path.join(
            annotations_save_dir, "annotations.csv"
        )
        annotation_df.to_csv(annotation_df_path, index=False)

        project = ClassificationProject(
            name=project_name,
            image_type=image_type,
            annotation_directories=["annotations"],
            annotation_df_path=os.path.relpath(
                annotation_df_path, project_dir
            ),
            data_directories=data_directories,
            mask_directories=mask_directories,
            display_mode=display_mode,
            classes=classes,
            project_dir=project_dir,
        )
        project.save()
        return project_dir

    def _run_other_creation(
        self,
        project_name,
        image_type,
        project_type,
        project_dir,
        data_directories,
        copy_data,
        status,
    ):
        os.makedirs(project_dir, exist_ok=True)
        annotations_save_dir = os.path.join(project_dir, "annotations")
        os.makedirs(annotations_save_dir, exist_ok=True)

        for data_dir in data_directories:
            if os.path.isdir(data_dir):
                d = convert_path_to_dir_name(data_dir)
                os.makedirs(
                    os.path.join(annotations_save_dir, d), exist_ok=True
                )

        annotation_directories = [
            os.path.relpath(os.path.join(annotations_save_dir, d), project_dir)
            for d in os.listdir(annotations_save_dir)
            if os.path.isdir(os.path.join(annotations_save_dir, d))
        ]

        if copy_data:
            local_data_dir = os.path.join(project_dir, "data")
            os.makedirs(local_data_dir, exist_ok=True)
            status.emit("Copying data...")
            data_directories = self._copy_data_directories_static(
                data_directories, local_data_dir, status
            )

        project = Project(
            name=project_name,
            image_type=image_type,
            project_type=project_type,
            annotation_directories=annotation_directories,
            data_directories=data_directories,
            project_dir=project_dir,
        )
        project.save()
        return project_dir

    @staticmethod
    def _copy_data_directories_static(
        data_directories, local_data_dir, status
    ):
        for data_dir in data_directories:
            if os.path.isdir(data_dir):
                dest_dir = os.path.join(
                    local_data_dir, convert_path_to_dir_name(data_dir)
                )
                if not os.path.exists(dest_dir):
                    status.emit(f"Copying {os.path.basename(data_dir)}...")
                    shutil.copytree(data_dir, dest_dir)

        return [
            os.path.join(local_data_dir, d)
            for d in os.listdir(local_data_dir)
            if os.path.isdir(os.path.join(local_data_dir, d))
        ]
