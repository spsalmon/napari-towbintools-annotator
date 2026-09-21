"""Cross-plane annotation propagation in the panoptic annotator."""

import numpy as np
import pandas as pd
import pytest
import tifffile

from napari_towbintools_annotator.panoptic_annotator import (
    PanopticAnnotatorWidget,
)
from napari_towbintools_annotator.project import PanopticProject

# Nucleus A spans planes 0-2, labelled 1 / 5 / 2 (independent per plane).
# Nucleus B sits alone on plane 1, labelled 9.
A_LABELS = {0: 1, 1: 5, 2: 2}
B_PLANE, B_LABEL = 1, 9


def _segmentation():
    seg = np.zeros((4, 20, 20), dtype=np.uint16)
    for z, label in A_LABELS.items():
        seg[z, 2:6, 2:6] = label
    seg[B_PLANE, 14:18, 14:18] = B_LABEL
    return seg


def _make_project(tmp_path, threshold=0.25, segmentation=None):
    project_dir = tmp_path / "proj"
    annotations_dir = project_dir / "annotations"
    annotations_dir.mkdir(parents=True)

    seg = _segmentation() if segmentation is None else segmentation
    ref = np.zeros(seg.shape, dtype=np.uint16)
    ref_path = tmp_path / "img.tif"
    seg_path = tmp_path / "img_seg.tif"
    tifffile.imwrite(str(ref_path), ref)
    tifffile.imwrite(str(seg_path), seg)

    pd.DataFrame(
        {
            "Reference": [str(ref_path)],
            "Segmentation": [str(seg_path)],
            "Annotation": [""],
        }
    ).to_csv(annotations_dir / "annotations.csv", index=False)

    return PanopticProject(
        name="p",
        image_type="zstack",
        annotation_directories=["annotations"],
        annotation_df_path="annotations/annotations.csv",
        data_directories=[str(tmp_path)],
        mask_directories=[str(tmp_path)],
        classes=["a", "b"],
        project_dir=str(project_dir),
        stitch_threshold=threshold,
    )


@pytest.fixture
def widget(tmp_path, viewer):
    yield PanopticAnnotatorWidget(viewer, _make_project(tmp_path))


def _planes(layer):
    return sorted(int(round(p[0])) for p in np.asarray(layer.data))


def _click(widget, z, y, x):
    """Placing a point programmatically fires the same event a click does."""
    widget._annotation_layer.add([[z, y, x]])


def test_click_propagates_to_every_plane_of_the_nucleus(widget):
    _click(widget, 1, 3, 3)

    assert _planes(widget._annotation_layer) == [0, 1, 2]


def test_propagated_dots_sit_on_the_nucleus_in_each_plane(widget):
    _click(widget, 1, 3, 3)

    seg = np.asarray(widget._segmentation_layer.data)
    for point in np.asarray(widget._annotation_layer.data):
        z, y, x = (int(round(c)) for c in point)
        assert seg[z, y, x] == A_LABELS[z]


def test_click_does_not_touch_a_different_nucleus(widget):
    _click(widget, 1, 15, 15)

    assert _planes(widget._annotation_layer) == [B_PLANE]


def test_reclick_recolours_whole_group_without_duplicating(widget):
    _click(widget, 1, 3, 3)
    widget.selected_class = "b"

    _click(widget, 0, 3, 3)

    layer = widget._annotation_layer
    assert _planes(layer) == [0, 1, 2]
    colors = np.asarray(layer.face_color)
    expected = np.asarray(widget.class_name_to_color["b"], dtype=float)
    for color in colors:
        np.testing.assert_allclose(color, expected, atol=1e-6)


def test_deleting_one_dot_leaves_the_others(widget):
    _click(widget, 1, 3, 3)
    layer = widget._annotation_layer

    plane_zero = [
        i
        for i, p in enumerate(np.asarray(layer.data))
        if int(round(p[0])) == 0
    ]
    layer.selected_data = set(plane_zero)
    layer.remove_selected()

    assert _planes(layer) == [1, 2]


def test_trimmed_plane_is_not_restored_by_a_later_click(widget):
    _click(widget, 1, 3, 3)
    layer = widget._annotation_layer

    plane_zero = [
        i
        for i, p in enumerate(np.asarray(layer.data))
        if int(round(p[0])) == 0
    ]
    layer.selected_data = set(plane_zero)
    layer.remove_selected()

    widget.selected_class = "b"
    _click(widget, 2, 3, 3)

    assert _planes(layer) == [1, 2]


def test_handler_ignores_events_raised_by_its_own_edits(widget):
    """Propagation rewrites the layer; those events must not re-enter it.

    Without the guard a re-entrant pass would recolour the group to whatever
    class is selected at the time, so a pending class change makes the
    difference observable.
    """
    _click(widget, 1, 3, 3)
    before = np.asarray(widget._annotation_layer.face_color).copy()
    widget.selected_class = "b"

    class _SelfInflictedEvent:
        action = "added"
        data_indices = (-1,)

    widget._propagating = True
    try:
        widget._on_points_changed(_SelfInflictedEvent())
    finally:
        widget._propagating = False

    np.testing.assert_allclose(
        np.asarray(widget._annotation_layer.face_color), before
    )


def test_clicking_background_leaves_no_point(widget):
    _click(widget, 0, 18, 2)

    assert len(widget._annotation_layer.data) == 0


def test_saving_records_original_labels_and_instance_id(widget, tmp_path):
    _click(widget, 1, 3, 3)
    widget.save_annotations()
    widget._save_master_sync()

    saved = pd.read_csv(tmp_path / "proj" / "annotations" / "img.csv")

    assert list(saved.columns) == [
        "Z",
        "Label",
        "ClassID",
        "Class",
        "InstanceID",
    ]
    assert len(saved) == 3
    by_plane = saved.set_index("Z")["Label"].to_dict()
    assert by_plane == A_LABELS
    assert saved["InstanceID"].nunique() == 1


def test_two_nuclei_get_different_instance_ids(widget, tmp_path):
    _click(widget, 1, 3, 3)
    _click(widget, 1, 15, 15)
    widget.save_annotations()
    widget._save_master_sync()

    saved = pd.read_csv(tmp_path / "proj" / "annotations" / "img.csv")

    assert saved["InstanceID"].nunique() == 2


def test_saved_annotations_reload_onto_every_plane(widget):
    _click(widget, 1, 3, 3)
    widget.save_annotations()
    widget._save_master_sync()

    widget._load_file()

    assert _planes(widget._annotation_layer) == [0, 1, 2]


def test_restitching_at_a_higher_threshold_splits_the_nucleus(
    tmp_path, viewer
):
    """A threshold the planes cannot meet stops them being one instance.

    The blocks step across planes so consecutive IoU is 1/3: linked at the
    0.25 default, separate once the threshold is raised past it.
    """
    seg = np.zeros((4, 20, 20), dtype=np.uint16)
    seg[0, 2:6, 2:6] = 1
    seg[1, 4:8, 2:6] = 5
    seg[2, 6:10, 2:6] = 2

    widget = PanopticAnnotatorWidget(
        viewer, _make_project(tmp_path, segmentation=seg)
    )
    _click(widget, 1, 5, 3)
    assert _planes(widget._annotation_layer) == [0, 1, 2]

    widget.stitch_threshold_spinbox.setValue(0.95)
    widget.restitch()

    widget._set_points([], [])
    _click(widget, 1, 5, 3)
    assert _planes(widget._annotation_layer) == [1]


def test_two_dimensional_project_is_unaffected(tmp_path, viewer):
    """2D panoptic annotation must behave exactly as before."""
    seg = np.zeros((20, 20), dtype=np.uint16)
    seg[2:6, 2:6] = 5
    project = _make_project(tmp_path, segmentation=seg)

    widget = PanopticAnnotatorWidget(viewer, project)
    assert widget._instance_index is None

    widget._annotation_layer.data = np.array([[3, 3]])
    widget._annotation_layer.face_color = np.array(
        [widget.class_id_to_color[0]], dtype=float
    )
    widget.save_annotations()
    widget._save_master_sync()

    saved = pd.read_csv(tmp_path / "proj" / "annotations" / "img.csv")
    assert list(saved.columns) == ["Label", "ClassID", "Class"]
    assert int(saved.loc[0, "Label"]) == 5


def _click_like_napari_06(widget, z, y, x):
    """Place a point the way napari 0.6.x's ``Points.add`` announces it.

    That version reports ``data_indices`` as one index per *coordinate* of
    the added point instead of one per added point, so a 3D click arrives as
    ``(-3, -2, -1)``. The event is emitted by hand because the napari in the
    test environment is a version that reports ``(-1,)``.
    """
    layer = widget._annotation_layer
    with layer.events.data.blocker():
        layer.data = np.append(np.asarray(layer.data), [[z, y, x]], axis=0)
    layer.events.data(
        value=layer.data,
        action="added",
        data_indices=tuple(np.arange(-3, 0)),
        vertex_indices=((),),
    )


def test_second_nucleus_does_not_wipe_the_first_on_napari_06(widget):
    """Misreported indices must never cost an annotation already placed."""
    _click_like_napari_06(widget, 1, 3, 3)
    assert _planes(widget._annotation_layer) == [0, 1, 2]

    _click_like_napari_06(widget, 1, 15, 15)

    layer = widget._annotation_layer
    assert _planes(layer) == [0, 1, 1, 2]
    seg = np.asarray(widget._segmentation_layer.data)
    labels = sorted(
        int(seg[tuple(int(round(c)) for c in point)])
        for point in np.asarray(layer.data)
    )
    assert labels == sorted([*A_LABELS.values(), B_LABEL])


@pytest.mark.parametrize(
    "data_indices",
    [(-1,), (-3, -2, -1), (0, 1, 2), (), None],
    ids=["one-per-point", "one-per-axis", "all-points", "empty", "missing"],
)
def test_added_points_are_found_without_trusting_the_event(
    widget, data_indices
):
    """Whatever the event claims, only the trailing point is new."""
    _click_like_napari_06(widget, 1, 3, 3)
    layer = widget._annotation_layer

    with layer.events.data.blocker():
        layer.data = np.append(np.asarray(layer.data), [[1, 15, 15]], axis=0)
    kwargs = {} if data_indices is None else {"data_indices": data_indices}
    layer.events.data(
        value=layer.data, action="added", vertex_indices=((),), **kwargs
    )

    assert _planes(layer) == [0, 1, 1, 2]


def test_background_click_only_drops_its_own_point(widget):
    """A miss removes the dot it created, not the ones already placed."""
    _click_like_napari_06(widget, 1, 3, 3)

    _click_like_napari_06(widget, 0, 18, 2)

    assert _planes(widget._annotation_layer) == [0, 1, 2]
