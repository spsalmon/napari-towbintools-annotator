"""Panoptic annotator widget: navigation, class selection, key bindings."""

import imageio
import numpy as np
import pandas as pd
import pytest
import tifffile
from napari.utils.colormaps.standardize_color import transform_color
from napari_align_annotator.panoptic_annotator import PanopticAnnotatorWidget
from napari_align_annotator.project import PanopticProject
from qtpy.QtGui import QCloseEvent

N_FILES = 3


def _make_project(tmp_path, n_files=N_FILES, annotated=(), classes=None):
    project_dir = tmp_path / "proj"
    annotations_dir = project_dir / "annotations"
    annotations_dir.mkdir(parents=True)

    rows = []
    for i in range(n_files):
        ref = tmp_path / f"img{i}.tif"
        seg = tmp_path / f"img{i}_seg.tif"
        tifffile.imwrite(str(ref), np.zeros((10, 10), dtype=np.uint8))
        mask = np.zeros((10, 10), dtype=np.uint16)
        mask[2:5, 2:5] = i + 1
        tifffile.imwrite(str(seg), mask)
        done = ""
        if i in annotated:
            done = str(annotations_dir / f"img{i}.csv")
            pd.DataFrame(
                {"Label": [i + 1], "ClassID": [1], "Class": ["b"]}
            ).to_csv(done, index=False)
        rows.append((str(ref), str(seg), done))

    pd.DataFrame(
        rows, columns=["Reference", "Segmentation", "Annotation"]
    ).to_csv(annotations_dir / "annotations.csv", index=False)

    return PanopticProject(
        name="p",
        image_type="multichannel",
        annotation_directories=["annotations"],
        annotation_df_path="annotations/annotations.csv",
        data_directories=[str(tmp_path)],
        mask_directories=[str(tmp_path)],
        classes=list(classes or ["a", "b", "c"]),
        project_dir=str(project_dir),
    )


def _key_names(viewer):
    return {str(binding) for binding in viewer.keymap}


@pytest.fixture
def widget(tmp_path, viewer):
    return PanopticAnnotatorWidget(viewer, _make_project(tmp_path))


def test_keys_are_bound_on_open(widget, viewer):
    assert {"Up", "Down", "J", "H", "S"} <= _key_names(viewer)


def test_closing_unbinds_keys_and_flushes_the_master(widget, viewer):
    widget.annotation_df.loc[0, "Annotation"] = "marker.csv"
    widget._pending_write = True

    widget.closeEvent(QCloseEvent())

    assert not {"Up", "Down", "J", "H", "S"} & _key_names(viewer)
    master = pd.read_csv(widget.annotation_df_path)
    assert master.loc[0, "Annotation"] == "marker.csv"


def test_class_keys_cycle_and_wrap(widget):
    assert widget.selected_class == "a"

    widget._cycle_class_down(None)
    assert widget.selected_class == "b"
    checked = widget.class_buttons.checkedButton()
    assert checked.text() == "b"

    widget._cycle_class_up(None)
    widget._cycle_class_up(None)
    assert widget.selected_class == "c"
    np.testing.assert_allclose(
        transform_color(widget._annotation_layer.current_face_color)[0],
        widget.class_name_to_color["c"],
        atol=1 / 255,
    )


def test_clicking_a_class_button_selects_it(widget):
    button = widget.class_buttons.buttons()[2]
    button.click()

    assert widget.selected_class == "c"


def test_navigation_keys_move_between_files(widget, viewer):
    widget._next_file_key(viewer)
    widget._next_file_key(viewer)
    widget._next_file_key(viewer)  # stays on the last file
    assert widget.current_file_idx == N_FILES - 1
    assert viewer.layers[0].name == f"img{N_FILES - 1}.tif"

    for _ in range(N_FILES + 1):
        widget._previous_file_key(viewer)
    assert widget.current_file_idx == 0
    assert widget.file_list_widget.currentRow() == 0


def test_navigation_buttons_move_between_files(widget):
    widget.next_button.click()
    assert widget.current_file_idx == 1

    widget.previous_button.click()
    assert widget.current_file_idx == 0


def test_loading_replaces_the_layers(widget, viewer):
    widget.next_file()

    assert [layer.name for layer in viewer.layers] == [
        "img1.tif",
        "img1_seg.tif",
        widget._annotation_layer.name,
    ]


def test_choosing_from_the_list_autosaves_and_loads(widget, viewer):
    widget._annotation_layer.data = np.array([[3, 3]])
    widget._annotation_layer.face_color = np.array(
        [widget.class_id_to_color[0]], dtype=float
    )

    widget.file_list_widget.setCurrentRow(2)
    widget.choose_file_from_list()
    widget._save_master_sync()

    assert widget.current_file_idx == 2
    assert viewer.layers[0].name == "img2.tif"
    saved = pd.read_csv(widget.annotation_df_path)
    assert saved.loc[0, "Annotation"].endswith("img0.csv")


def test_save_key_writes_and_marks_the_file_done(widget):
    widget._annotation_layer.data = np.array([[3, 3]])
    widget._annotation_layer.face_color = np.array(
        [widget.class_id_to_color[1]], dtype=float
    )

    widget._save_key(None)
    widget._save_master_sync()

    item = widget.file_list_widget.item(0)
    assert item.background().color().alpha() == 255
    saved = pd.read_csv(widget.annotation_df.loc[0, "Annotation"])
    assert saved.to_dict("list") == {
        "Label": [1],
        "ClassID": [1],
        "Class": ["b"],
    }


def test_opens_on_the_first_unannotated_file(tmp_path, viewer):
    project = _make_project(tmp_path, annotated=(0, 1))

    widget = PanopticAnnotatorWidget(viewer, project)

    assert widget.current_file_idx == 2
    assert widget.file_list_widget.item(0).background().color().alpha() == 255
    assert widget.file_list_widget.item(2).background().color().alpha() == 0


def test_fully_annotated_project_opens_at_the_start(tmp_path, viewer):
    project = _make_project(tmp_path, annotated=range(N_FILES))

    widget = PanopticAnnotatorWidget(viewer, project)

    assert widget.current_file_idx == 0
    # The saved annotation is replayed as a dot on its instance.
    assert len(widget._annotation_layer.data) == 1
    np.testing.assert_allclose(
        widget._annotation_layer.face_color[0],
        widget.class_id_to_color[1],
    )


def test_missing_annotation_file_is_ignored(tmp_path, viewer):
    project = _make_project(tmp_path)
    df = pd.read_csv(tmp_path / "proj" / "annotations" / "annotations.csv")
    df["Annotation"] = df["Annotation"].astype(object)
    df.loc[0, "Annotation"] = str(tmp_path / "gone.csv")
    df.to_csv(
        tmp_path / "proj" / "annotations" / "annotations.csv", index=False
    )

    widget = PanopticAnnotatorWidget(viewer, project)

    assert len(widget._annotation_layer.data) == 0


def test_empty_project_opens_without_layers(tmp_path, viewer):
    project = _make_project(tmp_path, n_files=0)

    widget = PanopticAnnotatorWidget(viewer, project)

    assert widget.file_list_widget.count() == 0
    assert len(viewer.layers) == 0
    # Nothing to act on: these must be no-ops.
    widget.next_file()
    widget.previous_file()
    widget.save_annotations()
    widget._cycle_class(1)
    widget.restitch()


def test_non_tiff_references_are_read(tmp_path, viewer):
    project = _make_project(tmp_path, n_files=1)
    png = tmp_path / "img0.png"
    imageio.imwrite(png, np.full((10, 10), 9, dtype=np.uint8))
    csv = tmp_path / "proj" / "annotations" / "annotations.csv"
    df = pd.read_csv(csv)
    df.loc[0, "Reference"] = str(png)
    df.to_csv(csv, index=False)

    PanopticAnnotatorWidget(viewer, project)

    assert viewer.layers[0].name == "img0.png"
    assert int(viewer.layers[0].data.max()) == 9


def test_saving_uses_the_reference_name(widget, tmp_path):
    widget.save_annotations()

    assert (tmp_path / "proj" / "annotations" / "img0.csv").is_file()


def test_a_stale_background_write_never_wins(widget, monkeypatch):
    from napari_align_annotator import panoptic_annotator

    pending = []

    class DeferredThread:
        def __init__(self, target, daemon=None):
            self.target = target

        def start(self):
            pending.append(self.target)

    monkeypatch.setattr(panoptic_annotator.threading, "Thread", DeferredThread)
    widget.save_annotations()
    widget.next_file()
    widget.save_annotations()
    assert len(pending) == 2

    # The OS schedules the older writer last.
    for write in reversed(pending):
        write()

    master = pd.read_csv(widget.annotation_df_path)
    assert master.loc[0, "Annotation"].endswith("img0.csv")
    assert master.loc[1, "Annotation"].endswith("img1.csv")
    assert widget._pending_write is False
