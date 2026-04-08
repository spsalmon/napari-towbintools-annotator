import os
import threading

import imageio
import pandas as pd
import tifffile
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QButtonGroup
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QListWidget
from qtpy.QtWidgets import QListWidgetItem
from qtpy.QtWidgets import QPushButton
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget

from .project import ClassificationProject

# Palette of visually distinct colors for class highlighting.
# Expressed as (background_hex, is_dark) — is_dark drives text color choice.
_CLASS_PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
]


def _read_image(path):
    try:
        return tifffile.imread(path)
    except Exception:
        return imageio.imread(path)


def _read_labels(path):
    return tifffile.imread(path)


class ClassificationAnnotatorWidget(QWidget):
    def __init__(
        self, napari_viewer, project: ClassificationProject, parent=None
    ):
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
        if "ImagePath" in self.annotation_df.columns:
            self.annotation_df["ImagePath"] = self.annotation_df[
                "ImagePath"
            ].astype(str)
        if "MaskPath" in self.annotation_df.columns:
            self.annotation_df["MaskPath"] = self.annotation_df[
                "MaskPath"
            ].astype(str)

        primary_col = (
            "ImagePath" if project.display_mode != "mask" else "MaskPath"
        )
        self.data_files = self.annotation_df[primary_col].tolist()

        # Map each class name to a stable color index.
        self._class_colors = {
            cls: _CLASS_PALETTE[i % len(_CLASS_PALETTE)]
            for i, cls in enumerate(project.classes)
        }

        self.annotation_df["Class"] = self.annotation_df["Class"].astype(str)

        self.file_list_widget = QListWidget()
        self._populate_file_list()

        self.current_file_idx = self._find_resume_index()
        if self.current_file_idx >= len(self.data_files):
            self.current_file_idx = 0

        self._image_layer = None
        self._mask_layer = None
        self._write_lock = threading.Lock()
        self._pending_write = False

        self.file_list_widget.setCurrentRow(self.current_file_idx)
        self._init_layers()
        self.file_list_widget.itemClicked.connect(self.choose_file_from_list)

        self.main_layout.addWidget(self.file_list_widget)

        # Current class status label
        self.class_status_label = QLabel("")
        self.class_status_label.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 4px;"
        )
        self.main_layout.addWidget(self.class_status_label)
        self._update_class_display(self.current_file_idx)

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

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self._save_sync)
        self.main_layout.addWidget(self.save_button)

    def _find_resume_index(self):
        if (
            "Class" not in self.annotation_df.columns
            or self.annotation_df.empty
        ):
            return 0

        classes = self.annotation_df["Class"].astype(str).str.strip()
        annotated = classes[
            (classes != "") & (classes != "nan") & (classes != "None")
        ]

        if annotated.empty:
            return 0

        return annotated.index[-1] + 1

    def _populate_file_list(self):
        self.file_list_widget.clear()
        for i, path in enumerate(self.data_files):
            item = QListWidgetItem(os.path.basename(path))
            self._apply_item_color(item, i)
            self.file_list_widget.addItem(item)

    def _apply_item_color(self, item, idx):
        class_name = str(self.annotation_df.loc[idx, "Class"]).strip()
        if (
            class_name in ("", "nan", "None")
            or class_name not in self._class_colors
        ):
            item.setBackground(QColor("transparent"))
            item.setForeground(QColor("white"))
            return

        bg = QColor(self._class_colors[class_name])
        # Use white or black text depending on luminance of background.
        luminance = 0.299 * bg.red() + 0.587 * bg.green() + 0.114 * bg.blue()
        item.setBackground(bg)
        item.setForeground(QColor("black" if luminance > 128 else "white"))

    def _update_class_display(self, idx):
        if idx < 0 or idx >= len(self.data_files):
            self.class_status_label.setText("")
            self.class_status_label.setStyleSheet(
                "font-weight: bold; font-size: 13px; padding: 4px;"
            )
            return

        class_name = str(self.annotation_df.loc[idx, "Class"]).strip()
        if class_name in ("", "nan", "None"):
            self.class_status_label.setText("Not annotated")
            self.class_status_label.setStyleSheet(
                "font-weight: bold; font-size: 13px; padding: 4px; color: gray;"
            )
        else:
            color = self._class_colors.get(class_name, "#888888")
            bg = QColor(color)
            luminance = (
                0.299 * bg.red() + 0.587 * bg.green() + 0.114 * bg.blue()
            )
            text_color = "black" if luminance > 128 else "white"
            self.class_status_label.setText(class_name)
            self.class_status_label.setStyleSheet(
                f"font-weight: bold; font-size: 13px; padding: 4px;"
                f"background-color: {color}; color: {text_color}; border-radius: 3px;"
            )

    def _init_layers(self):
        if self.current_file_idx < 0 or self.current_file_idx >= len(
            self.data_files
        ):
            return

        display_mode = self.project.display_mode
        row = self.annotation_df.iloc[self.current_file_idx]

        if (
            display_mode in ("image", "both")
            and "ImagePath" in self.annotation_df.columns
        ):
            data = _read_image(row["ImagePath"])
            self._image_layer = self.viewer.add_image(
                data,
                colormap="viridis",
                name=os.path.basename(row["ImagePath"]),
            )

        if (
            display_mode in ("mask", "both")
            and "MaskPath" in self.annotation_df.columns
        ):
            mask_path = row["MaskPath"]
            if pd.notna(mask_path) and mask_path not in ("", "nan", "None"):
                data = _read_labels(mask_path)
                self._mask_layer = self.viewer.add_labels(
                    data,
                    name=f"mask_{os.path.basename(mask_path)}",
                )

    def _load_file(self):
        if self.current_file_idx < 0 or self.current_file_idx >= len(
            self.data_files
        ):
            return

        display_mode = self.project.display_mode
        row = self.annotation_df.iloc[self.current_file_idx]

        if display_mode in ("image", "both") and self._image_layer is not None:
            self._image_layer.data = _read_image(row["ImagePath"])
            self._image_layer.name = os.path.basename(row["ImagePath"])

        if display_mode in ("mask", "both") and self._mask_layer is not None:
            mask_path = row["MaskPath"]
            if pd.notna(mask_path) and mask_path not in ("", "nan", "None"):
                self._mask_layer.data = _read_labels(mask_path)
                self._mask_layer.name = f"mask_{os.path.basename(mask_path)}"

        self._update_class_display(self.current_file_idx)

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
        if self.current_file_idx < 0 or self.current_file_idx >= len(
            self.data_files
        ):
            return

        class_name = button.text()
        self.annotation_df.loc[self.current_file_idx, "Class"] = class_name

        item = self.file_list_widget.item(self.current_file_idx)
        self._apply_item_color(item, self.current_file_idx)

        self.next_file()
        self._save_async()

    def ignore_file(self):
        if self.current_file_idx < 0 or self.current_file_idx >= len(
            self.data_files
        ):
            return

        self.annotation_df.drop(index=self.current_file_idx, inplace=True)
        self.annotation_df.reset_index(drop=True, inplace=True)

        self.data_files.pop(self.current_file_idx)
        self.file_list_widget.takeItem(self.current_file_idx)

        if self.data_files:
            self.current_file_idx = min(
                self.current_file_idx, len(self.data_files) - 1
            )
            self.file_list_widget.setCurrentRow(self.current_file_idx)
            self._load_file()

        self._save_async()

    def _save_sync(self):
        with self._write_lock:
            self._pending_write = False
            self.annotation_df.to_csv(self.annotation_df_path, index=False)

    def _save_async(self):
        snapshot = self.annotation_df.copy()
        path = self.annotation_df_path

        def write():
            with self._write_lock:
                self._pending_write = False
                snapshot.to_csv(path, index=False)

        self._pending_write = True
        threading.Thread(target=write, daemon=True).start()

    def closeEvent(self, event):
        if self._pending_write:
            self._save_sync()
        super().closeEvent(event)
