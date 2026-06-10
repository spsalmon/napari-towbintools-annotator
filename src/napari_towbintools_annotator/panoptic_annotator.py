import os
import threading

import numpy as np
from skimage.measure import regionprops

import pandas as pd
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QButtonGroup
from qtpy.QtWidgets import QHBoxLayout
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QListWidget
from qtpy.QtWidgets import QListWidgetItem
from qtpy.QtWidgets import QPushButton
from qtpy.QtWidgets import QRadioButton
from qtpy.QtWidgets import QVBoxLayout
from qtpy.QtWidgets import QWidget

from .colors import CLASS_PALETTE
from .colors import hex_to_rgba_float


def nearest_class_id(color, id_to_color):
    """Return the class id whose RGBA color is closest to ``color``."""
    target = np.asarray(color, dtype=float)
    best_id, best_dist = -1, float("inf")
    for class_id, class_color in id_to_color.items():
        dist = np.sum((target - np.asarray(class_color, dtype=float)) ** 2)
        if dist < best_dist:
            best_dist, best_id = dist, class_id
    return best_id


def points_to_rows(
    points, face_colors, label_data, id_to_color, id_to_name, plane_axis=None
):
    """Convert annotation points + colors into per-instance annotation rows.

    Each point is rounded to integer coordinates, used to read the label value
    under it, and its color is matched to the nearest class. Points outside the
    label array are skipped. In 3D (``plane_axis`` set) the first-axis index is
    recorded under that column name.
    """
    rows = []
    shape = label_data.shape
    for point, color in zip(points, face_colors):
        index = tuple(int(round(coord)) for coord in point)
        if len(index) != len(shape):
            continue
        if any(i < 0 or i >= s for i, s in zip(index, shape)):
            continue
        label_value = int(label_data[index])
        # Background (label 0) is not an annotatable instance; skip it.
        if label_value == 0:
            continue
        class_id = nearest_class_id(color, id_to_color)
        class_name = id_to_name.get(class_id, "unknown")
        if plane_axis is not None:
            row = {
                plane_axis: index[0],
                "Label": label_value,
                "ClassID": class_id,
                "Class": class_name,
            }
        else:
            row = {
                "Label": label_value,
                "ClassID": class_id,
                "Class": class_name,
            }
        rows.append(row)
    return rows


def _centroid(mask):
    props = regionprops(mask.astype(int))
    if not props:
        return None
    return props[0].centroid


def rows_to_points(annotations_df, label_data, id_to_color, plane_axis=None):
    """Convert annotation rows back into ``(point_coords, rgba)`` placements.

    For each row the label's centroid is used as the point location. Rows with
    an unknown class id, or whose label is absent from the (plane of the) label
    array, are skipped.
    """
    placements = []
    if plane_axis is not None:
        for plane in annotations_df[plane_axis].unique():
            plane = int(plane)
            if plane < 0 or plane >= label_data.shape[0]:
                continue
            plane_df = annotations_df[annotations_df[plane_axis] == plane]
            for _, row in plane_df.iterrows():
                color = id_to_color.get(int(row["ClassID"]))
                if color is None:
                    continue
                mask = label_data[plane] == int(row["Label"])
                if not mask.any():
                    continue
                centroid = _centroid(mask)
                if centroid is None:
                    continue
                placements.append(
                    (np.array([plane, centroid[0], centroid[1]]), color)
                )
    else:
        for _, row in annotations_df.iterrows():
            color = id_to_color.get(int(row["ClassID"]))
            if color is None:
                continue
            mask = label_data == int(row["Label"])
            if not mask.any():
                continue
            centroid = _centroid(mask)
            if centroid is None:
                continue
            placements.append(
                (np.array([centroid[0], centroid[1]]), color)
            )
    return placements


_PLANE_AXIS = "Z"
_DONE_COLOR = "#55A868"


class PanopticAnnotatorWidget(QWidget):
    def __init__(self, napari_viewer, project, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.project = project

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.project_label = QLabel(f"Project: {project.name}")
        self.main_layout.addWidget(self.project_label)

        self.annotation_df_path = os.path.join(
            project.project_dir, project.annotation_df_path
        )
        self.annotation_df = pd.read_csv(self.annotation_df_path)
        for col in ("Reference", "Segmentation", "Annotation"):
            if col in self.annotation_df.columns:
                self.annotation_df[col] = (
                    self.annotation_df[col].fillna("").astype(str)
                )
        self.reference_files = self.annotation_df["Reference"].tolist()

        # Class lookups; colors derived from palette by class index.
        self.classes = list(project.classes)
        self.class_id_to_name = dict(enumerate(self.classes))
        self.class_name_to_id = {c: i for i, c in enumerate(self.classes)}
        self.class_id_to_color = {
            i: hex_to_rgba_float(CLASS_PALETTE[i % len(CLASS_PALETTE)])
            for i in range(len(self.classes))
        }
        self.class_name_to_color = {
            c: self.class_id_to_color[i]
            for i, c in enumerate(self.classes)
        }
        self.selected_class = self.classes[0] if self.classes else None

        # Layer + write state.
        self._reference_layer = None
        self._segmentation_layer = None
        self._annotation_layer = None
        self._write_lock = threading.Lock()
        self._pending_write = False

        # File list.
        self.file_list_widget = QListWidget()
        self._populate_file_list()
        self.current_file_idx = self._find_resume_index()
        if self.current_file_idx >= len(self.reference_files):
            self.current_file_idx = 0
        self.file_list_widget.setCurrentRow(self.current_file_idx)
        self.file_list_widget.itemClicked.connect(self.choose_file_from_list)
        self.main_layout.addWidget(self.file_list_widget)

        # Navigation.
        nav_layout = QHBoxLayout()
        self.previous_button = QPushButton("Previous [H]")
        self.next_button = QPushButton("Next [J]")
        self.previous_button.clicked.connect(self.previous_file)
        self.next_button.clicked.connect(self.next_file)
        nav_layout.addWidget(self.previous_button)
        nav_layout.addWidget(self.next_button)
        self.main_layout.addLayout(nav_layout)

        # Class radio buttons.
        self.class_buttons_widget = QWidget()
        self.class_buttons_layout = QVBoxLayout()
        self.class_buttons_widget.setLayout(self.class_buttons_layout)
        self.class_buttons = QButtonGroup(self)
        for class_name in self.classes:
            button = QRadioButton(class_name)
            self._style_class_button(button, class_name)
            self.class_buttons.addButton(button)
            self.class_buttons_layout.addWidget(button)
            if class_name == self.selected_class:
                button.setChecked(True)
        self.class_buttons.buttonClicked.connect(self._on_class_button)
        self.main_layout.addWidget(self.class_buttons_widget)

        self.save_button = QPushButton("Save annotations [S]")
        self.save_button.clicked.connect(self.save_annotations)
        self.main_layout.addWidget(self.save_button)

        # Key bindings.
        self._bound_keys = {
            "Up": self._cycle_class_up,
            "Down": self._cycle_class_down,
            "j": self._next_file_key,
            "h": self._previous_file_key,
            "s": self._save_key,
        }
        for key, callback in self._bound_keys.items():
            self.viewer.bind_key(key, callback, overwrite=True)

        self._load_file()

    # ----- file list -----
    def _populate_file_list(self):
        self.file_list_widget.clear()
        for i, path in enumerate(self.reference_files):
            item = QListWidgetItem(os.path.basename(path))
            self._apply_item_color(item, i)
            self.file_list_widget.addItem(item)

    def _apply_item_color(self, item, idx):
        annotation = str(self.annotation_df.loc[idx, "Annotation"]).strip()
        if annotation in ("", "nan", "None"):
            item.setBackground(QColor("transparent"))
            item.setForeground(QColor("white"))
        else:
            item.setBackground(QColor(_DONE_COLOR))
            item.setForeground(QColor("white"))

    def _find_resume_index(self):
        annotated = self.annotation_df["Annotation"].astype(str).str.strip()
        for i, value in enumerate(annotated):
            if value in ("", "nan", "None"):
                return i
        return 0

    # ----- class selection -----
    def _style_class_button(self, button, class_name):
        color = CLASS_PALETTE[
            self.class_name_to_id[class_name] % len(CLASS_PALETTE)
        ]
        bg = QColor(color)
        luminance = 0.299 * bg.red() + 0.587 * bg.green() + 0.114 * bg.blue()
        text_color = "black" if luminance > 128 else "white"
        button.setStyleSheet(
            f"QRadioButton {{ background-color: {color}; "
            f"color: {text_color}; padding: 3px; border-radius: 3px; }}"
        )

    def _on_class_button(self, button):
        self.selected_class = button.text()
        self._update_point_color()

    def _cycle_class(self, delta):
        if not self.classes:
            return
        idx = (
            self.classes.index(self.selected_class) + delta
        ) % len(self.classes)
        self.selected_class = self.classes[idx]
        for button in self.class_buttons.buttons():
            if button.text() == self.selected_class:
                button.setChecked(True)
        self._update_point_color()

    def _update_point_color(self):
        if self._annotation_layer is None or self.selected_class is None:
            return
        self._annotation_layer.selected_data = set()
        self._annotation_layer.current_face_color = (
            self.class_name_to_color[self.selected_class]
        )

    # ----- file loading -----
    def _plane_axis(self):
        if self._segmentation_layer is None:
            return None
        return _PLANE_AXIS if self._segmentation_layer.ndim == 3 else None

    def _add_annotation_layer(self):
        ndim = self._segmentation_layer.ndim
        self._annotation_layer = self.viewer.add_points(
            np.zeros((0, ndim)), name="Annotations", ndim=ndim, size=10
        )
        self._update_point_color()

    def _replay_annotations(self, csv_path):
        df = pd.read_csv(csv_path)
        label_data = np.asarray(self._segmentation_layer.data)
        placements = rows_to_points(
            df, label_data, self.class_id_to_color, self._plane_axis()
        )
        if not placements:
            return
        coords = np.array([point for point, _ in placements])
        colors = np.array([color for _, color in placements], dtype=float)
        self._annotation_layer.data = coords
        self._annotation_layer.face_color = colors

    def _load_file(self):
        if not self.reference_files or not (
            0 <= self.current_file_idx < len(self.reference_files)
        ):
            return

        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()
        self._reference_layer = None
        self._segmentation_layer = None
        self._annotation_layer = None

        row = self.annotation_df.iloc[self.current_file_idx]
        reference_file = row["Reference"]
        segmentation_file = row["Segmentation"]
        annotation_file = str(row["Annotation"]).strip()

        self._reference_layer = self.viewer.open(reference_file)[-1]
        self._segmentation_layer = self.viewer.open(
            segmentation_file, layer_type="labels", opacity=0.5
        )[-1]
        self._add_annotation_layer()

        if annotation_file not in ("", "nan", "None") and os.path.isfile(
            annotation_file
        ):
            self._replay_annotations(annotation_file)

        self.viewer.reset_view()

    def choose_file_from_list(self):
        self.current_file_idx = self.file_list_widget.currentRow()
        self._load_file()

    def next_file(self):
        if not self.reference_files:
            return
        self.current_file_idx = min(
            self.current_file_idx + 1, len(self.reference_files) - 1
        )
        self.file_list_widget.setCurrentRow(self.current_file_idx)
        self._load_file()

    def previous_file(self):
        if not self.reference_files:
            return
        self.current_file_idx = max(self.current_file_idx - 1, 0)
        self.file_list_widget.setCurrentRow(self.current_file_idx)
        self._load_file()

    # ----- saving -----
    def save_annotations(self):
        if self._annotation_layer is None or self._segmentation_layer is None:
            return
        label_data = np.asarray(self._segmentation_layer.data)
        rows = points_to_rows(
            np.asarray(self._annotation_layer.data),
            np.asarray(self._annotation_layer.face_color),
            label_data,
            self.class_id_to_color,
            self.class_id_to_name,
            self._plane_axis(),
        )
        df = pd.DataFrame(rows)

        reference = self.annotation_df.loc[self.current_file_idx, "Reference"]
        name = os.path.splitext(os.path.basename(reference))[0]
        annotations_dir = os.path.dirname(self.annotation_df_path)
        out_path = os.path.join(annotations_dir, f"{name}.csv")
        df.to_csv(out_path, index=False)

        self.annotation_df.loc[self.current_file_idx, "Annotation"] = out_path
        item = self.file_list_widget.item(self.current_file_idx)
        self._apply_item_color(item, self.current_file_idx)
        self._save_master_async()

    def _save_master_sync(self):
        with self._write_lock:
            self._pending_write = False
            self.annotation_df.to_csv(self.annotation_df_path, index=False)

    def _save_master_async(self):
        snapshot = self.annotation_df.copy()
        path = self.annotation_df_path

        def write():
            with self._write_lock:
                self._pending_write = False
                snapshot.to_csv(path, index=False)

        self._pending_write = True
        threading.Thread(target=write, daemon=True).start()

    # ----- key callbacks (napari passes the viewer) -----
    def _cycle_class_up(self, viewer=None):
        self._cycle_class(-1)

    def _cycle_class_down(self, viewer=None):
        self._cycle_class(1)

    def _next_file_key(self, viewer=None):
        self.next_file()

    def _previous_file_key(self, viewer=None):
        self.previous_file()

    def _save_key(self, viewer=None):
        self.save_annotations()

    def closeEvent(self, event):
        for key in self._bound_keys:
            try:
                self.viewer.bind_key(key, None, overwrite=True)
            except Exception:
                pass
        if self._pending_write:
            self._save_master_sync()
        super().closeEvent(event)
