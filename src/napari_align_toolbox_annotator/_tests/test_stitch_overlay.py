"""Visual feedback for the z-stitch: overlay layer and count readout."""

import numpy as np
import pytest
from napari_align_annotator.panoptic_annotator import PanopticAnnotatorWidget
from napari_align_annotator.stitching import build_instance_index
from napari_align_annotator.stitching import instance_volume

from .test_panoptic_propagation import _make_project

_OVERLAY = "Stitched instances"


def _stepped_stack():
    """Blocks stepping across planes; consecutive IoU is 1/3."""
    seg = np.zeros((3, 20, 20), dtype=np.uint16)
    seg[0, 2:6, 2:6] = 1
    seg[1, 4:8, 2:6] = 5
    seg[2, 6:10, 2:6] = 2
    return seg


@pytest.fixture
def widget(tmp_path, viewer):
    yield PanopticAnnotatorWidget(
        viewer, _make_project(tmp_path, segmentation=_stepped_stack())
    )


def test_readout_reports_the_collapse_after_loading_a_file(widget):
    text = widget.stitch_readout_label.text()

    assert "3 objects" in text
    assert "1 instance" in text


def test_overlay_is_absent_until_the_box_is_ticked(widget):
    assert _OVERLAY not in widget.viewer.layers


def test_ticking_the_box_shows_one_id_per_nucleus_through_the_stack(widget):
    widget.show_stitched_checkbox.setChecked(True)

    volume = np.asarray(widget.viewer.layers[_OVERLAY].data)
    assert volume[0, 3, 3] == volume[1, 5, 3] == volume[2, 7, 3]
    assert volume[0, 18, 18] == 0


def test_unticking_the_box_removes_the_overlay(widget):
    widget.show_stitched_checkbox.setChecked(True)
    widget.show_stitched_checkbox.setChecked(False)

    assert _OVERLAY not in widget.viewer.layers


def test_overlay_sits_below_the_annotation_points(widget):
    widget.show_stitched_checkbox.setChecked(True)

    layers = widget.viewer.layers
    assert layers.index(layers[_OVERLAY]) < layers.index(
        widget._annotation_layer
    )


def test_restitching_at_a_higher_threshold_redraws_the_overlay(widget):
    widget.show_stitched_checkbox.setChecked(True)
    before = np.array(widget.viewer.layers[_OVERLAY].data)

    widget.stitch_threshold_spinbox.setValue(0.95)
    widget.restitch()

    after = np.asarray(widget.viewer.layers[_OVERLAY].data)
    assert before[0, 3, 3] == before[1, 5, 3]
    assert after[0, 3, 3] != after[1, 5, 3]


def test_restitching_updates_the_readout(widget):
    widget.stitch_threshold_spinbox.setValue(0.95)
    widget.restitch()

    assert "3 instances" in widget.stitch_readout_label.text()


def test_reloading_a_file_leaves_a_single_overlay_layer(widget):
    widget.show_stitched_checkbox.setChecked(True)
    widget._load_file()

    names = [layer.name for layer in widget.viewer.layers]
    assert names.count(_OVERLAY) == 1


def test_the_overlay_matches_the_index_it_was_built_from(widget):
    widget.show_stitched_checkbox.setChecked(True)

    expected = instance_volume(
        build_instance_index(_stepped_stack(), threshold=0.25)
    )
    assert np.array_equal(
        np.asarray(widget.viewer.layers[_OVERLAY].data), expected
    )


def test_two_dimensional_projects_get_no_stitch_feedback(tmp_path, viewer):
    seg = np.zeros((20, 20), dtype=np.uint16)
    seg[2:6, 2:6] = 5

    widget = PanopticAnnotatorWidget(
        viewer, _make_project(tmp_path, segmentation=seg)
    )
    assert not widget.stitch_widget.isVisible()
    assert _OVERLAY not in viewer.layers
