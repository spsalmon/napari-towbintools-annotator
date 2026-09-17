"""Whole-image class annotation widget."""

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from qtpy.QtGui import QCloseEvent

from napari_towbintools_annotator.classification_annotator import (
    ClassificationAnnotatorWidget,
)
from napari_towbintools_annotator.colors import CLASS_PALETTE
from napari_towbintools_annotator.project import ClassificationProject

# Real acquisition file names: spaces and commas must survive the CSV.
TEST_IMAGES = sorted((Path(__file__).parent / "test_images").glob("*.tiff"))[
    :3
]
CLASSES = ["alive", "dead"]


def _make_project(tmp_path, display_mode="image", classes=None):
    project_dir = tmp_path / "proj"
    annotations_dir = project_dir / "annotations"
    annotations_dir.mkdir(parents=True)

    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    columns = {}
    images, masks = [], []
    for i, source in enumerate(TEST_IMAGES):
        image = image_dir / source.name
        shutil.copy(source, image)
        images.append(str(image))
        mask = mask_dir / f"mask_{i}.tif"
        shape = tifffile.imread(source).shape
        tifffile.imwrite(str(mask), np.full(shape, i + 1, dtype=np.uint16))
        masks.append(str(mask))

    if display_mode != "mask":
        columns["ImagePath"] = images
    if display_mode != "image":
        columns["MaskPath"] = masks
    columns["Class"] = [np.nan] * len(images)
    pd.DataFrame(columns).to_csv(
        annotations_dir / "annotations.csv", index=False
    )

    return ClassificationProject(
        name="c",
        image_type="multichannel",
        annotation_directories=["annotations"],
        annotation_df_path="annotations/annotations.csv",
        project_dir=str(project_dir),
        classes=list(classes or CLASSES),
        data_directories=[str(image_dir)],
        mask_directories=[str(mask_dir)],
        display_mode=display_mode,
    )


def _master(project):
    return pd.read_csv(Path(project.project_dir) / project.annotation_df_path)


def _button(widget, name):
    return next(b for b in widget.class_buttons.buttons() if b.text() == name)


@pytest.fixture
def project(tmp_path):
    return _make_project(tmp_path)


@pytest.fixture
def widget(project, viewer):
    return ClassificationAnnotatorWidget(viewer, project)


def test_opens_on_the_first_image(widget, viewer):
    assert widget.current_file_idx == 0
    assert widget.file_list_widget.count() == len(TEST_IMAGES)
    assert widget.file_list_widget.item(0).text() == TEST_IMAGES[0].name
    assert [layer.name for layer in viewer.layers] == [TEST_IMAGES[0].name]
    assert widget.class_status_label.text() == "Not annotated"


def test_one_button_per_class(widget):
    assert [b.text() for b in widget.class_buttons.buttons()] == CLASSES


def test_assigning_a_class_records_it_and_advances(widget, project, viewer):
    widget.assign_class(_button(widget, "dead"))
    widget._save_sync()

    assert widget.current_file_idx == 1
    assert viewer.layers[0].name == TEST_IMAGES[1].name
    assert viewer.layers[0].data.shape == tifffile.imread(TEST_IMAGES[1]).shape
    master = _master(project)
    assert master.loc[0, "Class"] == "dead"
    assert master["Class"].isna().sum() == len(TEST_IMAGES) - 1
    # Paths with spaces and commas are read back intact.
    assert master["ImagePath"].tolist() == widget.data_files


def test_clicking_a_class_button_assigns_it(widget, project):
    _button(widget, "alive").click()
    widget._save_sync()

    assert _master(project).loc[0, "Class"] == "alive"


def test_assigned_file_is_coloured_by_class(widget):
    widget.assign_class(_button(widget, "dead"))

    item = widget.file_list_widget.item(0)
    assert item.background().color().name() == CLASS_PALETTE[1].lower()


def test_status_label_shows_the_class_of_the_current_file(widget):
    widget.assign_class(_button(widget, "alive"))
    widget.current_file_idx = 0
    widget._load_file()

    assert widget.class_status_label.text() == "alive"
    assert CLASS_PALETTE[0] in widget.class_status_label.styleSheet()


def test_status_label_is_blank_out_of_range(widget):
    widget._update_class_display(99)

    assert widget.class_status_label.text() == ""


def test_next_file_wraps_around(widget):
    for _ in TEST_IMAGES:
        widget.next_file()

    assert widget.current_file_idx == 0
    assert widget.file_list_widget.currentRow() == 0


def test_choosing_from_the_list_loads_that_file(widget, viewer):
    widget.file_list_widget.setCurrentRow(2)
    widget.choose_file_from_list()

    assert widget.current_file_idx == 2
    assert viewer.layers[0].name == TEST_IMAGES[2].name


def test_ignoring_a_file_drops_it_from_the_project(widget, project, viewer):
    widget.ignore_file()
    widget._save_sync()

    assert widget.file_list_widget.count() == len(TEST_IMAGES) - 1
    assert widget.data_files[0].endswith(TEST_IMAGES[1].name)
    assert viewer.layers[0].name == TEST_IMAGES[1].name
    master = _master(project)
    assert len(master) == len(TEST_IMAGES) - 1
    assert not master["ImagePath"].str.endswith(TEST_IMAGES[0].name).any()


def test_ignoring_the_last_file_steps_back(widget):
    widget.current_file_idx = len(TEST_IMAGES) - 1
    widget.ignore_file()

    assert widget.current_file_idx == len(TEST_IMAGES) - 2


def test_ignoring_every_file_leaves_an_empty_project(widget, project):
    for _ in TEST_IMAGES:
        widget.ignore_file()
    widget._save_sync()

    assert widget.data_files == []
    assert len(_master(project)) == 0
    # Nothing left to act on: these must be no-ops, not crashes.
    widget.assign_class(_button(widget, "alive"))
    widget.ignore_file()
    widget._load_file()


def test_reopening_resumes_after_the_last_annotated_file(project, viewer):
    first = ClassificationAnnotatorWidget(viewer, project)
    first.assign_class(_button(first, "alive"))
    first.assign_class(_button(first, "dead"))
    first._save_sync()
    viewer.layers.clear()

    reopened = ClassificationAnnotatorWidget(viewer, project)

    assert reopened.current_file_idx == 2
    assert reopened.file_list_widget.currentRow() == 2
    assert reopened.annotation_df.loc[1, "Class"] == "dead"


def test_fully_annotated_project_resumes_at_the_start(project, viewer):
    df = _master(project)
    df["Class"] = "alive"
    df.to_csv(
        Path(project.project_dir) / project.annotation_df_path, index=False
    )

    widget = ClassificationAnnotatorWidget(viewer, project)

    assert widget.current_file_idx == 0


def test_unknown_class_in_the_csv_is_left_uncoloured(project, viewer):
    df = _master(project)
    df["Class"] = df["Class"].astype(object)
    df.loc[0, "Class"] = "renamed-away"
    df.to_csv(
        Path(project.project_dir) / project.annotation_df_path, index=False
    )

    widget = ClassificationAnnotatorWidget(viewer, project)

    assert widget.file_list_widget.item(0).background().color().alpha() == 0
    widget.current_file_idx = 0
    widget._load_file()
    assert widget.class_status_label.text() == "renamed-away"


def test_async_save_writes_the_csv(widget, project):
    widget.annotation_df.loc[0, "Class"] = "alive"
    widget._save_async()
    # The writer clears the flag under the lock before writing, so taking
    # the lock once the flag is down waits for the write to finish.
    deadline = time.monotonic() + 10
    while widget._pending_write:
        assert time.monotonic() < deadline, "background write never ran"
        time.sleep(0.01)
    with widget._write_lock:
        pass

    assert _master(project).loc[0, "Class"] == "alive"


def test_closing_flushes_a_pending_write(widget, project):
    widget.annotation_df.loc[0, "Class"] = "dead"
    widget._pending_write = True

    widget.closeEvent(QCloseEvent())

    assert widget._pending_write is False
    assert _master(project).loc[0, "Class"] == "dead"


def test_mask_mode_shows_only_the_mask(tmp_path, viewer):
    project = _make_project(tmp_path, display_mode="mask")
    widget = ClassificationAnnotatorWidget(viewer, project)

    assert widget._image_layer is None
    assert [layer.name for layer in viewer.layers] == ["mask_mask_0.tif"]
    assert widget.file_list_widget.item(0).text() == "mask_0.tif"

    widget.next_file()
    assert viewer.layers[0].name == "mask_mask_1.tif"
    assert np.unique(viewer.layers[0].data).tolist() == [2]


def test_both_mode_overlays_mask_on_image(tmp_path, viewer):
    project = _make_project(tmp_path, display_mode="both")
    widget = ClassificationAnnotatorWidget(viewer, project)

    assert [layer.name for layer in viewer.layers] == [
        TEST_IMAGES[0].name,
        "mask_mask_0.tif",
    ]
    assert viewer.layers[0].data.shape == viewer.layers[1].data.shape

    widget.next_file()
    assert [layer.name for layer in viewer.layers] == [
        TEST_IMAGES[1].name,
        "mask_mask_1.tif",
    ]


def test_both_mode_tolerates_a_missing_mask(tmp_path, viewer):
    project = _make_project(tmp_path, display_mode="both")
    csv = Path(project.project_dir) / project.annotation_df_path
    df = pd.read_csv(csv)
    df.loc[0, "MaskPath"] = np.nan
    df.to_csv(csv, index=False)

    widget = ClassificationAnnotatorWidget(viewer, project)

    assert widget._mask_layer is None
    assert [layer.name for layer in viewer.layers] == [TEST_IMAGES[0].name]


def test_non_tiff_images_are_read(tmp_path, viewer):
    import imageio

    project = _make_project(tmp_path)
    png = tmp_path / "images" / "plain.png"
    imageio.imwrite(png, np.full((8, 8), 7, dtype=np.uint8))
    csv = Path(project.project_dir) / project.annotation_df_path
    pd.DataFrame({"ImagePath": [str(png)], "Class": [np.nan]}).to_csv(
        csv, index=False
    )

    ClassificationAnnotatorWidget(viewer, project)

    assert viewer.layers[0].data.shape == (8, 8)
    assert int(viewer.layers[0].data.max()) == 7


def test_many_classes_reuse_the_palette(tmp_path, viewer):
    classes = [f"c{i}" for i in range(len(CLASS_PALETTE) + 2)]
    project = _make_project(tmp_path, classes=classes)
    widget = ClassificationAnnotatorWidget(viewer, project)

    widget.assign_class(_button(widget, classes[-1]))

    item = widget.file_list_widget.item(0)
    assert item.background().color().name() == CLASS_PALETTE[1].lower()


class _DeferredThread:
    """Stands in for threading.Thread so a test decides when writers run."""

    pending = []

    def __init__(self, target, daemon=None):
        self.target = target

    def start(self):
        _DeferredThread.pending.append(self.target)


@pytest.fixture
def deferred_writes(monkeypatch):
    from napari_towbintools_annotator import classification_annotator

    _DeferredThread.pending = []
    monkeypatch.setattr(
        classification_annotator.threading, "Thread", _DeferredThread
    )
    return _DeferredThread.pending


def test_a_stale_background_write_never_wins(widget, project, deferred_writes):
    widget.assign_class(_button(widget, "alive"))
    widget.assign_class(_button(widget, "dead"))
    assert len(deferred_writes) == 2

    # The OS schedules the older writer last.
    for write in reversed(deferred_writes):
        write()

    master = _master(project)
    assert master.loc[:1, "Class"].tolist() == ["alive", "dead"]


def test_a_background_write_older_than_a_sync_save_is_dropped(
    widget, project, deferred_writes
):
    widget.assign_class(_button(widget, "alive"))
    widget.annotation_df.loc[1, "Class"] = "dead"
    widget._save_sync()

    deferred_writes[0]()

    assert _master(project).loc[1, "Class"] == "dead"
