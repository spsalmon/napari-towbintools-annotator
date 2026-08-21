"""Rendering an instance index back into a displayable volume."""

import numpy as np
import pytest

from napari_towbintools_annotator.stitching import (
    build_instance_index,
    instance_counts,
    instance_volume,
)


def _stepped_stack():
    """Three blocks stepping across planes; consecutive IoU is 1/3."""
    masks = np.zeros((3, 20, 20), dtype=np.uint16)
    masks[0, 2:6, 2:6] = 1
    masks[1, 4:8, 2:6] = 5
    masks[2, 6:10, 2:6] = 2
    return masks


def test_instance_volume_gives_one_nucleus_one_id_across_planes():
    masks = np.zeros((3, 10, 10), dtype=np.uint16)
    masks[0, 2:6, 2:6] = 1
    masks[1, 2:6, 2:6] = 7
    masks[2, 2:6, 2:6] = 3

    volume = instance_volume(build_instance_index(masks))

    ids = {int(volume[z, 3, 3]) for z in range(3)}
    assert len(ids) == 1
    assert 0 not in ids


def test_instance_volume_keeps_unlinked_nuclei_apart():
    masks = np.zeros((2, 20, 20), dtype=np.uint16)
    masks[0, 1:5, 1:5] = 1
    masks[0, 12:16, 12:16] = 2
    masks[1, 1:5, 1:5] = 1
    masks[1, 12:16, 12:16] = 2

    volume = instance_volume(build_instance_index(masks))

    assert volume[0, 3, 3] != volume[0, 14, 14]


def test_instance_volume_leaves_background_at_zero():
    volume = instance_volume(build_instance_index(_stepped_stack()))

    assert volume[0, 18, 18] == 0
    assert np.count_nonzero(volume) == np.count_nonzero(_stepped_stack())


def test_instance_volume_matches_the_stack_shape_and_is_int32():
    volume = instance_volume(build_instance_index(_stepped_stack()))

    assert volume.shape == (3, 20, 20)
    assert volume.dtype == np.int32


def test_instance_volume_follows_the_threshold():
    linked = instance_volume(build_instance_index(_stepped_stack(), 0.25))
    split = instance_volume(build_instance_index(_stepped_stack(), 0.95))

    assert linked[0, 3, 3] == linked[1, 5, 3]
    assert split[0, 3, 3] != split[1, 5, 3]


def test_instance_volume_needs_the_masks_it_was_built_from():
    index = build_instance_index(_stepped_stack())
    index.masks = None

    with pytest.raises(ValueError):
        instance_volume(index)


def test_instance_counts_reports_plane_objects_and_instances():
    index = build_instance_index(_stepped_stack(), 0.25)

    assert instance_counts(index) == (3, 1)


def test_instance_counts_splits_when_the_threshold_is_raised():
    index = build_instance_index(_stepped_stack(), 0.95)

    assert instance_counts(index) == (3, 3)
