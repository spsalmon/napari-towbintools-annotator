import datetime
import os
import shutil

import numpy as np
import pandas as pd
from napari_guitils.gui_structures import VHGroup
from natsort import natsorted
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from .project import Project, ClassificationProject
from pathlib import Path

class ClassificationAnnotatorWidget(QWidget):
    def __init__(self, napari_viewer, project, parent=None):
        super().__init__(parent=parent)

        self.viewer = napari_viewer
        self.project = project

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.project_label = QLabel(f"Project: {self.project.name}")
        self.main_layout.addWidget(self.project_label)

        self.annotation_df_path = os.path.join(
            project.project_dir, project.annotation_df_path
        )

        self.annotation_df = pd.read_csv(self.annotation_df_path)
        self.annotation_df["ImagePath"] = self.annotation_df["ImagePath"].astype(str)

        self.data_files = self.annotation_df["ImagePath"].tolist()

        self.file_list_widget = QListWidget()
        self.file_list_widget.addItems([os.path.basename(f) for f in self.data_files])

        self.current_file_idx = self._find_resume_index()
        self.annotation_df["Class"] = self.annotation_df["Class"].astype(str)

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

    def _find_resume_index(self):
        if "Class" not in self.annotation_df.columns or self.annotation_df.empty:
            return 0

        classes = self.annotation_df["Class"].astype(str).str.strip()
        annotated = classes[(classes != "") & (classes != "nan") & (classes != "None")]

        if annotated.empty:
            return 0

        return annotated.index[-1] + 1

    def _load_file(self):
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()

        if self.current_file_idx < 0 or self.current_file_idx >= len(self.data_files):
            return

        data_file = self.data_files[self.current_file_idx]
        class_name = self.annotation_df.loc[self.current_file_idx, "Class"]

        self.viewer.open(data_file, colormap="viridis", name=os.path.basename(data_file))

        if not (pd.isna(class_name) or class_name in ("", "nan", "None")):
            image_shape = np.shape(self.viewer.layers[-1].data)
            box = np.array([
                [0, 0],
                [0, image_shape[1]],
                [image_shape[0], image_shape[1]],
                [image_shape[0], 0],
            ])

            self.viewer.add_shapes(
                box,
                shape_type="polygon",
                features={"class": class_name},
                edge_width=3,
                text={
                    "string": "class",
                    "size": 20,
                    "anchor": "upper_left",
                    "color": "white",
                },
                face_color="transparent",
                edge_color="red",
                name="class name",
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
            return

        self.annotation_df.loc[self.current_file_idx, "Class"] = button.text()
        self.annotation_df.to_csv(self.annotation_df_path, index=False)
        self.next_file()

    def ignore_file(self):
        if self.current_file_idx < 0 or self.current_file_idx >= len(self.data_files):
            return

        self.annotation_df.drop(index=self.current_file_idx, inplace=True)
        self.annotation_df.reset_index(drop=True, inplace=True)
        self.annotation_df.to_csv(self.annotation_df_path, index=False)

        self.data_files.pop(self.current_file_idx)
        self.file_list_widget.takeItem(self.current_file_idx)

        if self.data_files:
            self.current_file_idx = min(self.current_file_idx, len(self.data_files) - 1)
            self.file_list_widget.setCurrentRow(self.current_file_idx)
            self._load_file()