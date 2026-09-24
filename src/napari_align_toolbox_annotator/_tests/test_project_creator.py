"""Project creation widget and the top-level plugin widget."""

import time

import numpy as np
import pandas as pd
import pytest
import tifffile
from napari_align_annotator.classification_annotator import (
    ClassificationAnnotatorWidget,
)
from napari_align_annotator.panoptic_annotator import PanopticAnnotatorWidget
from napari_align_annotator.project import ClassificationProject
from napari_align_annotator.project import Project
from napari_align_annotator.project_creator import ProjectCreatorWidget
from napari_align_annotator.project_creator import alignAnnotatorWidget
from napari_align_annotator.project_creator import convert_path_to_dir_name
from napari_align_annotator.project_creator import create_annotator_widget
from qtpy.QtWidgets import QFileDialog
from qtpy.QtWidgets import QLabel
from qtpy.QtWidgets import QListWidget
from qtpy.QtWidgets import QMessageBox

# The creation back ends only use ``self`` for a staticmethod, so the class
# stands in for a widget instance.
_NO_WIDGET = ProjectCreatorWidget


class _Status:
    def __init__(self):
        self.messages = []

    def emit(self, message):
        self.messages.append(message)


def _write_images(directory, names, value=0):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        tifffile.imwrite(
            str(directory / name), np.full((6, 6), value, dtype=np.uint16)
        )
    return directory


@pytest.fixture
def errors(monkeypatch):
    """Capture validation errors instead of opening a modal dialog."""
    shown = []

    def fake_exec(box):
        shown.append(box.text())
        return 0

    # raising=True: the widget calls exec_(), so it must exist on this Qt.
    monkeypatch.setattr(QMessageBox, "exec_", fake_exec, raising=True)
    return shown


@pytest.fixture
def creator(viewer):
    return ProjectCreatorWidget(viewer)


def _wait_for_worker(qapp, widget, timeout=10):
    assert widget._worker.wait(timeout * 1000)
    deadline = time.monotonic() + timeout
    while widget.create_button.isEnabled() is False:
        qapp.processEvents()
        assert time.monotonic() < deadline, "worker result never delivered"


# ----- pure helpers -----


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/mnt/data/images/", "mnt_data_images"),
        ("C:\\Users\\me\\scans", "C__Users_me_scans"),
        ("relative/dir", "relative_dir"),
    ],
)
def test_convert_path_to_dir_name(path, expected):
    assert convert_path_to_dir_name(path) == expected


def test_create_annotator_widget_rejects_unknown_types(viewer):
    project = Project("p", "zstack", "keypoint", [], [], "proj")

    with pytest.raises(NotImplementedError, match="keypoint"):
        create_annotator_widget(viewer, project)


def test_add_directory_to_a_multi_list(tmp_path):
    display, dirs = QListWidget(), []

    ProjectCreatorWidget._add_directory_to_list(
        str(tmp_path), display, dirs, multiple=True
    )
    ProjectCreatorWidget._add_directory_to_list(
        str(tmp_path), display, dirs, multiple=True
    )

    assert dirs == [str(tmp_path)] * 2
    assert display.count() == 2


def test_add_directory_to_a_single_slot_replaces_it(tmp_path):
    display, dirs = QLabel(), ["old"]
    (tmp_path / "new").mkdir()

    ProjectCreatorWidget._add_directory_to_list(
        str(tmp_path / "new"), display, dirs, multiple=False
    )

    assert dirs == [str(tmp_path / "new")]
    assert display.text() == str(tmp_path / "new")


@pytest.mark.parametrize("path", ["", "does/not/exist"])
def test_cancelled_or_bogus_directory_is_ignored(path):
    display, dirs = QListWidget(), []

    ProjectCreatorWidget._add_directory_to_list(path, display, dirs, True)

    assert dirs == []
    assert display.count() == 0


def test_remove_selected_directories(tmp_path):
    display = QListWidget()
    display.setSelectionMode(QListWidget.MultiSelection)
    dirs = ["a", "b", "c"]
    display.addItems(dirs)
    display.item(1).setSelected(True)

    ProjectCreatorWidget._remove_selected_directories(display, dirs)

    assert dirs == ["a", "c"]
    assert [display.item(i).text() for i in range(display.count())] == [
        "a",
        "c",
    ]


# ----- creator widget state -----


def test_directory_buttons_record_the_chosen_directory(
    creator, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
    )

    for selector in (
        creator.data_selection_widget,
        creator.mask_dir_selector_widget,
        creator.project_dir_selection_layout,
    ):
        selector.findChildren(type(creator.create_button))[0].click()

    assert creator.data_directories == [str(tmp_path)]
    assert creator.mask_directories == [str(tmp_path)]
    assert creator.project_dir == [str(tmp_path)]


def test_classes_can_be_added_once_and_removed(creator):
    for name in ("alive", "  alive ", "dead", ""):
        creator.class_input.setText(name)
        creator.add_class_button.click()

    assert creator._get_classes() == ["alive", "dead"]

    creator.classes_list.item(0).setSelected(True)
    creator.remove_class_button.click()
    assert creator._get_classes() == ["dead"]


def test_selected_types_are_reported(creator):
    assert creator._get_selected_image_type() == "multichannel"
    assert creator._get_selected_project_type() == "classification"
    assert creator._get_display_mode() == "image"

    creator.image_type_zstack.setChecked(True)
    creator.project_type_panoptic.setChecked(True)
    creator.display_mode_both.setChecked(True)
    assert creator._get_selected_image_type() == "zstack"
    assert creator._get_selected_project_type() == "panoptic"
    assert creator._get_display_mode() == "both"

    creator.image_type_time_series.setChecked(True)
    creator.project_type_keypoint.setChecked(True)
    creator.display_mode_mask.setChecked(True)
    assert creator._get_selected_image_type() == "time_series"
    assert creator._get_selected_project_type() == "keypoint"
    assert creator._get_display_mode() == "mask"


@pytest.mark.parametrize(
    ("mode", "images_visible", "masks_visible"),
    [("image", True, False), ("mask", False, True), ("both", True, True)],
)
def test_display_mode_picks_the_directory_selectors(
    creator, mode, images_visible, masks_visible
):
    getattr(creator, f"display_mode_{mode}").click()

    assert creator.data_selection_widget.isVisibleTo(creator) is images_visible
    assert (
        creator.mask_dir_selector_widget.isVisibleTo(creator) is masks_visible
    )


def test_keypoint_projects_cannot_be_selected(creator):
    # Keypoint annotation is not implemented yet: the option is greyed out
    # so a user cannot create a project that could never be reopened.
    assert not creator.project_type_keypoint.isEnabled()

    creator.project_type_keypoint.click()

    assert not creator.project_type_keypoint.isChecked()
    assert creator._get_selected_project_type() == "classification"


def test_panoptic_projects_need_both_directories(creator):
    creator.display_mode_mask.click()
    creator.project_type_panoptic.click()

    assert creator.data_selection_widget.isVisibleTo(creator)
    assert creator.mask_dir_selector_widget.isVisibleTo(creator)


def test_project_name_defaults_to_a_dated_name(creator):
    assert creator.project_name_input.text().endswith("_project")


# ----- validation -----


def test_empty_name_is_rejected(creator, errors):
    creator.project_name_input.setText("   ")
    creator.create_project()

    assert errors == ["Please enter a project name."]
    assert not hasattr(creator, "_worker")


def test_mask_mode_without_mask_directories_is_rejected(creator, errors):
    creator.display_mode_mask.click()
    creator.create_project()

    assert errors == ["No mask directories selected."]


def test_panoptic_without_masks_is_rejected(creator, errors):
    creator.project_type_panoptic.click()
    creator.create_project()

    assert errors == ["No segmentation (mask) directories selected."]


def test_panoptic_without_classes_is_rejected(creator, errors, tmp_path):
    creator.project_type_panoptic.click()
    creator.mask_directories.append(str(tmp_path))
    creator.create_project()

    assert errors == ["No classes defined."]


# ----- creation back ends -----


def test_classification_creation_lists_images_in_natural_order(tmp_path):
    images = _write_images(tmp_path / "images", ["img10.tif", "img2.tif"])
    project_dir = tmp_path / "proj"
    status = _Status()

    returned = ProjectCreatorWidget._run_classification_creation(
        _NO_WIDGET,
        "proj",
        "multichannel",
        "image",
        str(project_dir),
        [str(images)],
        [],
        ["a", "b"],
        False,
        status,
    )

    assert returned == str(project_dir)
    master = pd.read_csv(project_dir / "annotations" / "annotations.csv")
    assert list(master.columns) == ["ImagePath", "Class"]
    assert [p.rsplit("img", 1)[1] for p in master["ImagePath"]] == [
        "2.tif",
        "10.tif",
    ]
    assert master["Class"].isna().all()
    loaded = Project.load(str(project_dir))
    assert isinstance(loaded, ClassificationProject)
    assert loaded.annotation_df_path == str(
        (project_dir / "annotations" / "annotations.csv").relative_to(
            project_dir
        )
    )
    assert "Writing annotation file..." in status.messages


def test_classification_creation_in_mask_mode(tmp_path):
    masks = _write_images(tmp_path / "masks", ["m1.tif", "m2.tif"])
    project_dir = tmp_path / "proj"

    ProjectCreatorWidget._run_classification_creation(
        _NO_WIDGET,
        "proj",
        "multichannel",
        "mask",
        str(project_dir),
        [],
        [str(masks)],
        ["a"],
        False,
        _Status(),
    )

    master = pd.read_csv(project_dir / "annotations" / "annotations.csv")
    assert list(master.columns) == ["MaskPath", "Class"]
    assert len(master) == 2
    assert Project.load(str(project_dir)).display_mode == "mask"


def test_classification_creation_pairs_images_and_masks(tmp_path):
    images = _write_images(tmp_path / "images", ["a.tif", "b.tif"])
    masks = _write_images(tmp_path / "masks", ["a_m.tif", "b_m.tif"])
    project_dir = tmp_path / "proj"

    ProjectCreatorWidget._run_classification_creation(
        _NO_WIDGET,
        "proj",
        "multichannel",
        "both",
        str(project_dir),
        [str(images)],
        [str(masks)],
        ["a"],
        False,
        _Status(),
    )

    master = pd.read_csv(project_dir / "annotations" / "annotations.csv")
    assert list(master.columns) == ["ImagePath", "Class", "MaskPath"]
    assert master["MaskPath"].str.endswith(("a_m.tif", "b_m.tif")).all()


def test_classification_creation_rejects_unpaired_masks(tmp_path):
    images = _write_images(tmp_path / "images", ["a.tif", "b.tif"])
    masks = _write_images(tmp_path / "masks", ["a_m.tif"])

    with pytest.raises(ValueError, match="2 image files vs 1 mask files"):
        ProjectCreatorWidget._run_classification_creation(
            _NO_WIDGET,
            "proj",
            "multichannel",
            "both",
            str(tmp_path / "proj"),
            [str(images)],
            [str(masks)],
            ["a"],
            False,
            _Status(),
        )


def test_classification_creation_can_copy_the_data(tmp_path):
    images = _write_images(tmp_path / "images", ["a.tif"])
    project_dir = tmp_path / "proj"

    ProjectCreatorWidget._run_classification_creation(
        _NO_WIDGET,
        "proj",
        "multichannel",
        "image",
        str(project_dir),
        [str(images)],
        [],
        ["a"],
        True,
        _Status(),
    )

    master = pd.read_csv(project_dir / "annotations" / "annotations.csv")
    copied = master.loc[0, "ImagePath"]
    assert copied.startswith(str(project_dir / "data"))
    assert (project_dir / "data").is_dir()
    assert Project.load(str(project_dir)).data_directories == [
        str(project_dir / "data" / convert_path_to_dir_name(str(images)))
    ]


def test_copying_the_same_directory_twice_is_a_no_op(tmp_path):
    images = _write_images(tmp_path / "images", ["a.tif"])
    local = tmp_path / "local"
    local.mkdir()
    status = _Status()

    first = ProjectCreatorWidget._copy_data_directories_static(
        [str(images), str(tmp_path / "missing")], str(local), status
    )
    second = ProjectCreatorWidget._copy_data_directories_static(
        [str(images)], str(local), status
    )

    assert first == second
    assert len(first) == 1
    assert len([m for m in status.messages if m.startswith("Copying")]) == 1


def test_other_project_types_cannot_be_saved_yet(tmp_path):
    """Keypoint projects have no Project subclass, so creation fails."""
    images = _write_images(tmp_path / "images", ["a.tif"])

    with pytest.raises(NotImplementedError):
        ProjectCreatorWidget._run_other_creation(
            _NO_WIDGET,
            "proj",
            "multichannel",
            "keypoint",
            str(tmp_path / "proj"),
            [str(images)],
            False,
            _Status(),
        )

    assert (
        tmp_path
        / "proj"
        / "annotations"
        / convert_path_to_dir_name(str(images))
    ).is_dir()


# ----- end to end through the plugin widget -----


@pytest.fixture
def plugin(viewer):
    return alignAnnotatorWidget(viewer)


def test_create_button_toggles_the_creator(plugin):
    plugin.create_button.click()
    assert isinstance(plugin.project_creation_widget, ProjectCreatorWidget)
    assert not plugin.initial_button_widget.isVisibleTo(plugin)

    plugin.toggle_create_project()
    assert plugin.project_creation_widget is None
    assert plugin.initial_button_widget.isVisibleTo(plugin)


def test_cancel_returns_to_the_start_buttons(plugin, qapp):
    plugin.toggle_create_project()
    plugin.project_creation_widget.cancel_button.click()
    qapp.processEvents()

    assert plugin.project_creation_widget is None
    assert plugin.initial_button_widget.isVisibleTo(plugin)


def test_creating_a_classification_project_opens_it(
    plugin, qapp, tmp_path, viewer
):
    images = _write_images(tmp_path / "images", ["a.tif", "b.tif"], value=3)
    plugin.toggle_create_project()
    creator = plugin.project_creation_widget
    creator.project_name_input.setText("demo")
    creator.data_directories.append(str(images))
    creator.project_dir.append(str(tmp_path))
    creator.class_input.setText("alive")
    creator.add_class_button.click()

    creator.create_button.click()
    assert creator.status_label.text() == "Creating project..."
    _wait_for_worker(qapp, creator)

    assert plugin.project_creation_widget is None
    assert isinstance(plugin.annotator_widget, ClassificationAnnotatorWidget)
    assert plugin.annotator_widget.project.classes == ["alive"]
    assert (tmp_path / "demo" / "project.yaml").is_file()
    assert [layer.name for layer in viewer.layers] == ["a.tif"]


def test_creating_a_panoptic_project_opens_it(plugin, qapp, tmp_path):
    images = _write_images(tmp_path / "images", ["a.tif"])
    masks = _write_images(tmp_path / "masks", ["a.tif"], value=1)
    plugin.toggle_create_project()
    creator = plugin.project_creation_widget
    creator.project_name_input.setText("pan")
    creator.project_type_panoptic.click()
    creator.stitch_threshold_spinbox.setValue(0.5)
    creator.data_directories.append(str(images))
    creator.mask_directories.append(str(masks))
    creator.project_dir.append(str(tmp_path))
    creator.class_input.setText("nucleus")
    creator.add_class_button.click()

    creator.create_project()
    _wait_for_worker(qapp, creator)

    assert isinstance(plugin.annotator_widget, PanopticAnnotatorWidget)
    assert Project.load(str(tmp_path / "pan")).stitch_threshold == 0.5


def test_creation_failure_is_reported_and_the_form_stays(
    plugin, qapp, tmp_path, errors
):
    images = _write_images(tmp_path / "images", ["a.tif", "b.tif"])
    masks = _write_images(tmp_path / "masks", ["a.tif"])
    plugin.toggle_create_project()
    creator = plugin.project_creation_widget
    creator.project_type_panoptic.click()
    creator.data_directories.append(str(images))
    creator.mask_directories.append(str(masks))
    creator.project_dir.append(str(tmp_path))
    creator.class_input.setText("nucleus")
    creator.add_class_button.click()

    creator.create_project()
    _wait_for_worker(qapp, creator)

    assert len(errors) == 1
    assert "File count mismatch" in errors[0]
    assert plugin.project_creation_widget is creator
    assert creator.cancel_button.isEnabled()
    assert creator.status_label.text() == ""


def test_load_button_opens_the_chosen_project(
    plugin, tmp_path, monkeypatch, viewer
):
    images = _write_images(tmp_path / "images", ["a.tif"])
    ProjectCreatorWidget._run_classification_creation(
        _NO_WIDGET,
        "proj",
        "multichannel",
        "image",
        str(tmp_path / "proj"),
        [str(images)],
        [],
        ["a"],
        False,
        _Status(),
    )
    monkeypatch.setattr(
        QFileDialog,
        "getExistingDirectory",
        lambda *a, **k: str(tmp_path / "proj"),
    )

    plugin.load_button.click()

    assert isinstance(plugin.annotator_widget, ClassificationAnnotatorWidget)
    assert not plugin.initial_button_widget.isVisibleTo(plugin)
    assert len(viewer.layers) == 1


def test_cancelled_load_changes_nothing(plugin, monkeypatch, capsys):
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **k: ""
    )

    plugin.load_button.click()

    assert not hasattr(plugin, "annotator_widget")
    assert plugin.initial_button_widget.isVisibleTo(plugin)
    assert "Invalid project directory" in capsys.readouterr().out
