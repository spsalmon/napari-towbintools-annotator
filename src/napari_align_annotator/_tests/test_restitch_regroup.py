"""Re-stitching regroups placed dots under the new instance identity."""

import numpy as np
import pytest

from napari_align_annotator.panoptic_annotator import PanopticAnnotatorWidget
from napari_align_annotator.panoptic_annotator import majority_class_colors
from napari_align_annotator.stitching import build_instance_index

from .test_panoptic_propagation import _make_project

_RED = (1.0, 0.0, 0.0, 1.0)
_BLUE = (0.0, 0.0, 1.0, 1.0)
_ID_TO_COLOR = {0: _RED, 1: _BLUE}


def _stepped_stack():
    """Blocks stepping across planes; consecutive IoU is 1/3."""
    seg = np.zeros((3, 20, 20), dtype=np.uint16)
    seg[0, 2:6, 2:6] = 1
    seg[1, 4:8, 2:6] = 5
    seg[2, 6:10, 2:6] = 2
    return seg


def _two_nuclei():
    seg = np.zeros((2, 20, 20), dtype=np.uint16)
    seg[:, 2:6, 2:6] = 1
    seg[:, 14:18, 14:18] = 2
    return seg


def test_the_class_used_most_wins_the_whole_instance():
    """Two dots red, one blue, one nucleus: the nucleus goes red."""
    index = build_instance_index(_stepped_stack(), threshold=0.25)
    points = [(0, 3, 3), (1, 5, 3), (2, 7, 3)]

    colors = majority_class_colors(
        points, [_RED, _BLUE, _RED], index, _ID_TO_COLOR
    )

    assert np.allclose(colors, [_RED, _RED, _RED])


def test_separate_instances_keep_their_own_classes():
    index = build_instance_index(_two_nuclei(), threshold=0.25)
    points = [(0, 3, 3), (1, 3, 3), (0, 15, 15), (1, 15, 15)]

    colors = majority_class_colors(
        points, [_RED, _RED, _BLUE, _BLUE], index, _ID_TO_COLOR
    )

    assert np.allclose(colors, [_RED, _RED, _BLUE, _BLUE])


def test_a_tie_goes_to_the_most_recently_placed_dot():
    index = build_instance_index(_stepped_stack(), threshold=0.25)
    points = [(0, 3, 3), (1, 5, 3)]

    colors = majority_class_colors(points, [_RED, _BLUE], index, _ID_TO_COLOR)

    assert np.allclose(colors, [_BLUE, _BLUE])


def test_a_dot_on_background_keeps_its_own_color():
    index = build_instance_index(_stepped_stack(), threshold=0.25)

    colors = majority_class_colors([(0, 18, 18)], [_BLUE], index, _ID_TO_COLOR)

    assert np.allclose(colors, [_BLUE])


def test_one_color_is_returned_per_point():
    index = build_instance_index(_stepped_stack(), threshold=0.25)
    points = [(0, 3, 3), (1, 5, 3), (2, 7, 3)]

    colors = majority_class_colors(
        points, [_RED, _BLUE, _RED], index, _ID_TO_COLOR
    )

    assert np.asarray(colors).shape == (3, 4)


# ----- through the widget -----


def _select(widget, class_name):
    widget.selected_class = class_name
    widget._update_point_color()


def _click(widget, z, y, x):
    widget._annotation_layer.add([[z, y, x]])


def _planes(layer):
    return sorted(int(round(p[0])) for p in np.asarray(layer.data))


@pytest.fixture
def widget(tmp_path, viewer):
    """Annotator on the stepped stack, starting with the blocks unlinked."""
    yield PanopticAnnotatorWidget(
        viewer,
        _make_project(tmp_path, threshold=0.95, segmentation=_stepped_stack()),
    )


def _annotate_blocks_a_b_a(widget):
    _select(widget, "a")
    _click(widget, 0, 3, 3)
    _select(widget, "b")
    _click(widget, 1, 5, 3)
    _select(widget, "a")
    _click(widget, 2, 7, 3)


def test_merging_gives_every_dot_the_majority_class(widget):
    _annotate_blocks_a_b_a(widget)

    widget.stitch_threshold_spinbox.setValue(0.25)
    widget.restitch()

    colors = np.asarray(widget._annotation_layer.face_color)
    assert np.allclose(colors, widget.class_name_to_color["a"])


def test_restitching_leaves_the_dots_where_they_are(widget):
    _annotate_blocks_a_b_a(widget)
    before = np.array(widget._annotation_layer.data)

    widget.stitch_threshold_spinbox.setValue(0.25)
    widget.restitch()

    assert np.allclose(widget._annotation_layer.data, before)
    assert _planes(widget._annotation_layer) == [0, 1, 2]


def test_restitching_with_no_dots_placed_is_harmless(widget):
    widget.stitch_threshold_spinbox.setValue(0.25)
    widget.restitch()

    assert len(widget._annotation_layer.data) == 0
