import contextlib
import os
import threading

import imageio
import numpy as np
import pandas as pd
import tifffile
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from .colors import CLASS_PALETTE, hex_to_rgba_float
from .stitching import (
    centroid_table,
    get_instance_index,
    instance_counts,
    instance_volume,
)


def _read_array(path):
    try:
        return tifffile.imread(path)
    except Exception:  # noqa: BLE001
        return imageio.imread(path)

def channel_axis_first(image, mask_shape):
    """Move the axes around so that the mask Z sliders matches the image's Z slider.
    """
    mask_shape = tuple(mask_shape)
    if image.ndim != len(mask_shape) + 1:
        return image
    return image.swapaxes(0, 1)


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
    points,
    face_colors,
    label_data,
    id_to_color,
    id_to_name,
    plane_axis=None,
    instance_index=None,
):
    """Convert annotation points + colors into per-instance annotation rows.

    Each point is rounded to integer coordinates, used to read the label value
    under it, and its color is matched to the nearest class. Points outside the
    label array are skipped. In 3D (``plane_axis`` set) the first-axis index is
    recorded under that column name.

    ``Label`` is always the value read from the segmentation on disk. When an
    ``instance_index`` is supplied, an ``InstanceID`` column additionally
    records which stitched 3D object the row belongs to, so downstream code can
    group planes without re-deriving identity.
    """
    rows = []
    shape = label_data.shape
    for point, color in zip(points, face_colors, strict=False):
        index = tuple(int(round(coord)) for coord in point)
        if len(index) != len(shape):
            continue
        if any(i < 0 or i >= s for i, s in zip(index, shape, strict=False)):
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
        if instance_index is not None:
            row["InstanceID"] = instance_index.plane_label_to_instance.get(
                (index[0], label_value), -1
            )
        rows.append(row)
    return rows


def rows_to_points(
    annotations_df, label_data, id_to_color, plane_axis=None, centroids=None
):
    """Convert annotation rows back into ``(point_coords, rgba)`` placements.

    For each row the label's centroid is used as the point location. Rows with
    an unknown class id, or whose label has no centroid, are skipped.

    ``centroids`` is a ``{(z, label): (y, x)}`` table; 2D data is keyed on
    plane 0. It is computed from ``label_data`` when not supplied. Passing the
    table in matters: the caller usually already holds one, and rebuilding it
    per file is the difference between a redraw and a stall.
    """
    if centroids is None:
        stack = label_data if plane_axis is not None else label_data[None]
        centroids = centroid_table(stack)

    placements = []
    for _, row in annotations_df.iterrows():
        color = id_to_color.get(int(row["ClassID"]))
        if color is None:
            continue
        plane = int(row[plane_axis]) if plane_axis is not None else 0
        centroid = centroids.get((plane, int(row["Label"])))
        if centroid is None:
            continue
        if plane_axis is not None:
            placements.append(
                (np.array([plane, centroid[0], centroid[1]]), color)
            )
        else:
            placements.append((np.array([centroid[0], centroid[1]]), color))
    return placements


_PLANE_AXIS = "Z"
_DONE_COLOR = "#55A868"
_OVERLAY_LAYER_NAME = "Stitched instances"


def _plural(count, noun):
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


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

        # Cross-plane identity for the current z-stack (None in 2D). Set in
        # _load_file; _propagating guards the points handler against the
        # edits it makes itself.
        self._instance_index = None
        self._propagating = False
        self._stitch_overlay_layer = None
        self._stitch_cache_dir = os.path.join(
            os.path.dirname(
                os.path.join(project.project_dir, project.annotation_df_path)
            ),
            ".stitch_cache",
        )

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

        # Z-stitching controls; only meaningful for 3D segmentations.
        self.stitch_widget = QWidget()
        stitch_layout = QVBoxLayout()
        self.stitch_widget.setLayout(stitch_layout)
        stitch_controls = QHBoxLayout()
        stitch_layout.addLayout(stitch_controls)
        stitch_controls.addWidget(QLabel("Stitch IoU"))
        self.stitch_threshold_spinbox = QDoubleSpinBox()
        self.stitch_threshold_spinbox.setRange(0.05, 0.95)
        self.stitch_threshold_spinbox.setSingleStep(0.05)
        self.stitch_threshold_spinbox.setValue(
            getattr(project, "stitch_threshold", 0.25)
        )
        stitch_controls.addWidget(self.stitch_threshold_spinbox)
        self.restitch_button = QPushButton("Re-stitch")
        self.restitch_button.clicked.connect(self.restitch)
        stitch_controls.addWidget(self.restitch_button)
        self.show_stitched_checkbox = QCheckBox("Show stitched")
        self.show_stitched_checkbox.setToolTip(
            "Display-only overlay: one colour per nucleus through the "
            "stack. The segmentation on disk is never modified."
        )
        self.show_stitched_checkbox.toggled.connect(self._on_show_stitched)
        stitch_controls.addWidget(self.show_stitched_checkbox)
        self.stitch_readout_label = QLabel("")
        stitch_layout.addWidget(self.stitch_readout_label)
        self.stitch_widget.setVisible(False)
        self.main_layout.addWidget(self.stitch_widget)

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
        self._annotation_layer.events.data.connect(self._on_points_changed)
        self._update_point_color()

    # ----- cross-plane propagation -----
    def _build_instance_index(self):
        """Recover cross-plane nucleus identity for the current stack."""
        if self._segmentation_layer is None or (
            self._segmentation_layer.ndim != 3
        ):
            self._instance_index = None
            self._refresh_stitch_feedback()
            return
        row = self.annotation_df.iloc[self.current_file_idx]
        self._instance_index = get_instance_index(
            row["Segmentation"],
            np.asarray(self._segmentation_layer.data),
            self.stitch_threshold_spinbox.value(),
            self._stitch_cache_dir,
        )
        self._refresh_stitch_feedback()

    def _centroids(self):
        """Centroid table matching the current segmentation's dimensionality."""
        if self._instance_index is not None:
            return self._instance_index.centroids
        if self._segmentation_layer is None:
            return {}
        return centroid_table(
            np.asarray(self._segmentation_layer.data)[None]
        )

    def _instance_of_point(self, point):
        return self._instance_index.instance_at(
            *(int(round(coord)) for coord in point)
        )

    def _on_points_changed(self, event):
        """Propagate a freshly placed point across the nucleus it landed on.

        Only additions are acted on. Deletions are deliberately left alone so
        that removing a dot trims an over-eager stitch, and moves are ignored
        because saving resolves whatever label a point finally sits on.
        """
        if self._propagating or self._instance_index is None:
            return
        action = getattr(event, "action", None)
        if getattr(action, "value", action) != "added":
            return

        layer = self._annotation_layer
        data = np.asarray(layer.data)
        if len(data) == 0:
            return

        # The added points are the trailing ones; a click adds exactly one.
        indices = getattr(event, "data_indices", (-1,))
        added = sorted({int(i) % len(data) for i in indices})
        if not added:
            return
        clicked = data[added[-1]]
        instance = self._instance_of_point(clicked)

        keep = np.array(
            [i for i in range(len(data)) if i not in set(added)], dtype=int
        )
        if instance is None:
            # Landed on background: it can never produce a row, so drop it
            # rather than leave a dot that looks annotated.
            self._set_points(data[keep], np.asarray(layer.face_color)[keep])
            return

        self._rebuild_group(instance, int(round(clicked[0])), keep)

    def _rebuild_group(self, instance, clicked_plane, keep):
        """Redraw one nucleus's dots in the selected class.

        Target planes start from those already annotated, so a dot the user
        deleted stays deleted; a nucleus annotated for the first time gets a
        dot on every plane it occupies.
        """
        layer = self._annotation_layer
        data = np.asarray(layer.data)
        colors = np.asarray(layer.face_color)

        others, other_colors, annotated_planes = [], [], set()
        for i in keep:
            if self._instance_of_point(data[i]) == instance:
                annotated_planes.add(int(round(data[i][0])))
            else:
                others.append(data[i])
                other_colors.append(colors[i])

        planes = self._instance_index.planes_of(instance)
        targets = annotated_planes or set(planes)
        targets = sorted(targets | {clicked_plane})

        color = self.class_name_to_color[self.selected_class]
        for plane in targets:
            label = planes.get(plane)
            if label is None:
                continue
            centroid = self._instance_index.centroid(plane, label)
            if centroid is None:
                continue
            others.append(np.array([plane, centroid[0], centroid[1]]))
            other_colors.append(color)

        self._set_points(others, other_colors)

    def _set_points(self, points, colors):
        """Replace the layer's contents without re-entering the handler."""
        ndim = self._segmentation_layer.ndim
        self._propagating = True
        try:
            layer = self._annotation_layer
            if len(points) == 0:
                layer.data = np.zeros((0, ndim))
            else:
                layer.data = np.asarray(points, dtype=float)
                layer.face_color = np.asarray(colors, dtype=float)
            layer.selected_data = set()
        finally:
            self._propagating = False
        self._update_point_color()

    # ----- stitch feedback -----
    def _on_show_stitched(self, _checked):
        self._refresh_stitch_feedback()

    def _remove_stitch_overlay(self):
        layer = self._stitch_overlay_layer
        self._stitch_overlay_layer = None
        if layer is not None and layer in self.viewer.layers:
            self.viewer.layers.remove(layer)

    def _refresh_stitch_feedback(self):
        """Report what the current stitch did, in counts and in colour.

        The overlay is a display-only volume of instance ids. The
        segmentation layer keeps the values read from disk, because those
        are what ``points_to_rows`` writes to the ``Label`` column.
        """
        self._remove_stitch_overlay()
        if self._instance_index is None:
            self.stitch_readout_label.setText("")
            return

        objects, instances = instance_counts(self._instance_index)
        self.stitch_readout_label.setText(
            f"{_plural(objects, 'object')} \u2192 "
            f"{_plural(instances, 'instance')} "
            f"@ IoU {self._instance_index.threshold:.2f}"
        )

        if not self.show_stitched_checkbox.isChecked():
            return
        layer = self.viewer.add_labels(
            instance_volume(self._instance_index),
            name=_OVERLAY_LAYER_NAME,
            opacity=0.6,
        )
        self._stitch_overlay_layer = layer
        # Keep it under the points, which would otherwise be hidden by it.
        self.viewer.layers.move(
            self.viewer.layers.index(layer),
            self.viewer.layers.index(self._segmentation_layer) + 1,
        )

    def restitch(self):
        """Rebuild identity at the current threshold, keeping placed dots."""
        self._build_instance_index()

    def _replay_annotations(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            return
        if df.empty:
            return
        label_data = np.asarray(self._segmentation_layer.data)
        placements = rows_to_points(
            df,
            label_data,
            self.class_id_to_color,
            self._plane_axis(),
            centroids=self._centroids(),
        )
        if not placements:
            return
        coords = np.array([point for point, _ in placements])
        colors = np.array([color for _, color in placements], dtype=float)
        self._set_points(coords, colors)

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
        self._stitch_overlay_layer = None

        row = self.annotation_df.iloc[self.current_file_idx]
        reference_file = row["Reference"]
        segmentation_file = row["Segmentation"]
        annotation_file = str(row["Annotation"]).strip()

        segmentation = _read_array(segmentation_file)

        reference = channel_axis_first(
            _read_array(reference_file), segmentation.shape
        )

        self._reference_layer = self.viewer.add_image(
            reference, name=os.path.basename(reference_file)
        )
        self._segmentation_layer = self.viewer.add_labels(
            segmentation, name=os.path.basename(segmentation_file), opacity=0.5
        )
        self._build_instance_index()
        self.stitch_widget.setVisible(self._instance_index is not None)
        self._add_annotation_layer()

        if annotation_file not in ("", "nan", "None") and os.path.isfile(
            annotation_file
        ):
            self._replay_annotations(annotation_file)

        self.viewer.reset_view()

    def _autosave_current_file(self):
        """Persist the current file's annotations before navigating away.

        Files with no placed points are skipped so untouched files are not
        marked as done (which would also break resume-on-open). The explicit
        Save button still writes empty annotations when the user wants to.
        """
        if (
            self._annotation_layer is None
            or self._segmentation_layer is None
            or len(self._annotation_layer.data) == 0
        ):
            return
        self.save_annotations()

    def choose_file_from_list(self):
        self._autosave_current_file()
        self.current_file_idx = self.file_list_widget.currentRow()
        self._load_file()

    def next_file(self):
        if not self.reference_files:
            return
        self._autosave_current_file()
        self.current_file_idx = min(
            self.current_file_idx + 1, len(self.reference_files) - 1
        )
        self.file_list_widget.setCurrentRow(self.current_file_idx)
        self._load_file()

    def previous_file(self):
        if not self.reference_files:
            return
        self._autosave_current_file()
        self.current_file_idx = max(self.current_file_idx - 1, 0)
        self.file_list_widget.setCurrentRow(self.current_file_idx)
        self._load_file()

    # ----- saving -----
    def save_annotations(self):
        if self._annotation_layer is None or self._segmentation_layer is None:
            return
        plane_axis = self._plane_axis()
        label_data = np.asarray(self._segmentation_layer.data)
        rows = points_to_rows(
            np.asarray(self._annotation_layer.data),
            np.asarray(self._annotation_layer.face_color),
            label_data,
            self.class_id_to_color,
            self.class_id_to_name,
            plane_axis,
            instance_index=self._instance_index,
        )
        # InstanceID is appended last so positional readers of the older
        # column layout keep working.
        columns = (
            ([plane_axis] if plane_axis is not None else [])
            + ["Label", "ClassID", "Class"]
            + (["InstanceID"] if self._instance_index is not None else [])
        )
        df = pd.DataFrame(rows, columns=columns)

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
            with contextlib.suppress(Exception):
                self.viewer.bind_key(key, None, overwrite=True)
        if self._pending_write:
            self._save_master_sync()
        super().closeEvent(event)
