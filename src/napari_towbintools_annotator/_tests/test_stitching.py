import numpy as np

from napari_towbintools_annotator.stitching import (
    build_instance_index,
    centroid_table,
    stitch_planes,
)


def _block(masks, z, label, y0, y1, x0, x1):
    masks[z, y0:y1, x0:x1] = label


def test_stitch_planes_links_overlapping_labels_across_planes():
    """The same nucleus, labelled differently in each plane, becomes one id."""
    masks = np.zeros((3, 10, 10), dtype=np.uint16)
    _block(masks, 0, 1, 2, 6, 2, 6)
    _block(masks, 1, 7, 2, 6, 2, 6)
    _block(masks, 2, 3, 2, 6, 2, 6)

    stitched = stitch_planes(masks)

    ids = {int(stitched[z, 3, 3]) for z in range(3)}
    assert len(ids) == 1
    assert 0 not in ids


def test_stitch_planes_keeps_non_overlapping_nuclei_distinct():
    masks = np.zeros((2, 20, 20), dtype=np.uint16)
    _block(masks, 0, 1, 1, 5, 1, 5)
    _block(masks, 0, 2, 12, 16, 12, 16)
    _block(masks, 1, 1, 1, 5, 1, 5)
    _block(masks, 1, 2, 12, 16, 12, 16)

    stitched = stitch_planes(masks)

    assert stitched[0, 3, 3] == stitched[1, 3, 3]
    assert stitched[0, 14, 14] == stitched[1, 14, 14]
    assert stitched[0, 3, 3] != stitched[0, 14, 14]


def _weakly_overlapping():
    """Two blocks overlapping by an IoU of 1/3."""
    masks = np.zeros((2, 12, 12), dtype=np.uint16)
    _block(masks, 0, 1, 2, 6, 2, 6)
    _block(masks, 1, 1, 4, 8, 2, 6)
    return masks


def test_stitch_planes_merges_below_threshold():
    stitched = stitch_planes(_weakly_overlapping(), threshold=0.25)
    assert stitched[0, 3, 3] == stitched[1, 5, 3]


def test_stitch_planes_splits_above_threshold():
    stitched = stitch_planes(_weakly_overlapping(), threshold=0.5)
    assert stitched[0, 3, 3] != stitched[1, 5, 3]


def test_stitch_planes_handles_sparse_label_values():
    """Large, non-contiguous label values must not blow up the IoU matrix."""
    masks = np.zeros((2, 10, 10), dtype=np.uint16)
    _block(masks, 0, 60000, 2, 6, 2, 6)
    _block(masks, 1, 40000, 2, 6, 2, 6)

    stitched = stitch_planes(masks)

    assert stitched[0, 3, 3] == stitched[1, 3, 3]
    assert stitched.max() == 1


def test_stitch_planes_starts_new_id_after_empty_plane():
    masks = np.zeros((3, 10, 10), dtype=np.uint16)
    _block(masks, 0, 1, 2, 6, 2, 6)
    # plane 1 deliberately empty
    _block(masks, 2, 1, 2, 6, 2, 6)

    stitched = stitch_planes(masks)

    assert stitched[1].max() == 0
    assert stitched[0, 3, 3] != stitched[2, 3, 3]


def test_stitch_planes_survives_label_counts_that_overflow_uint16():
    """With ~300 objects/plane the IoU pair index exceeds uint16 range.

    The pair index is ``label_a * (n_b + 1) + label_b``, so a uint16 mask must
    be widened before that multiply or it wraps silently and mis-stitches.
    """
    n_objects = 300
    masks = np.zeros((2, 40, 40), dtype=np.uint16)
    coords = [(i // 40, i % 40) for i in range(n_objects)]
    for label, (y, x) in enumerate(coords, start=1):
        masks[0, y, x] = label
        masks[1, y, x] = label
    assert n_objects * (n_objects + 1) > np.iinfo(np.uint16).max

    stitched = stitch_planes(masks)

    # Identical planes: every object keeps its identity, none are merged.
    assert stitched[0].max() == n_objects
    np.testing.assert_array_equal(stitched[0], stitched[1])


def test_stitch_planes_rejects_non_3d_input():
    import pytest

    with pytest.raises(ValueError, match="3D"):
        stitch_planes(np.zeros((10, 10), dtype=np.uint16))


# ----- centroid_table -----


def test_centroid_table_keys_every_plane_label_pair():
    masks = np.zeros((2, 10, 10), dtype=np.uint16)
    _block(masks, 0, 4, 2, 6, 2, 6)
    _block(masks, 1, 9, 1, 3, 7, 9)

    table = centroid_table(masks)

    assert set(table) == {(0, 4), (1, 9)}
    np.testing.assert_allclose(table[(0, 4)], (3.5, 3.5))
    np.testing.assert_allclose(table[(1, 9)], (1.5, 7.5))


def test_centroid_table_matches_regionprops():
    """Pinned against skimage so the bbox optimisation cannot drift."""
    from skimage.measure import regionprops

    rng = np.random.default_rng(3)
    masks = np.zeros((4, 40, 40), dtype=np.uint16)
    for z in range(4):
        for label in range(1, 6):
            cy, cx = rng.integers(5, 35, 2)
            masks[z, cy - 3 : cy + 3, cx - 3 : cx + 3] = label

    table = centroid_table(masks)

    for z, plane in enumerate(masks):
        for prop in regionprops(plane.astype(int)):
            np.testing.assert_allclose(
                table[(z, prop.label)], prop.centroid, atol=1e-9
            )


def test_centroid_table_ignores_background_and_empty_planes():
    masks = np.zeros((2, 8, 8), dtype=np.uint16)
    _block(masks, 1, 2, 3, 5, 3, 5)

    table = centroid_table(masks)

    assert set(table) == {(1, 2)}


def _three_plane_nucleus():
    """One nucleus through 3 planes, labelled 1 / 7 / 3, plus a loner."""
    masks = np.zeros((3, 20, 20), dtype=np.uint16)
    _block(masks, 0, 1, 2, 6, 2, 6)
    _block(masks, 1, 7, 2, 6, 2, 6)
    _block(masks, 2, 3, 2, 6, 2, 6)
    _block(masks, 1, 8, 14, 18, 14, 18)
    return masks


def test_instance_index_groups_original_labels_by_instance():
    index = build_instance_index(_three_plane_nucleus())

    shared = index.plane_label_to_instance[(0, 1)]
    assert index.plane_label_to_instance[(1, 7)] == shared
    assert index.plane_label_to_instance[(2, 3)] == shared
    assert index.plane_label_to_instance[(1, 8)] != shared


def test_instance_index_planes_of_returns_original_labels_by_plane():
    index = build_instance_index(_three_plane_nucleus())
    shared = index.plane_label_to_instance[(0, 1)]

    assert index.planes_of(shared) == {0: 1, 1: 7, 2: 3}


def test_instance_index_instance_at_resolves_a_click():
    index = build_instance_index(_three_plane_nucleus())

    assert index.instance_at(1, 3, 3) == index.plane_label_to_instance[(0, 1)]
    assert (
        index.instance_at(1, 15, 15) == index.plane_label_to_instance[(1, 8)]
    )


def test_instance_index_instance_at_returns_none_off_object():
    index = build_instance_index(_three_plane_nucleus())

    assert index.instance_at(0, 10, 10) is None  # background
    assert index.instance_at(0, 500, 500) is None  # outside the plane
    assert index.instance_at(99, 3, 3) is None  # outside the stack


def test_instance_index_exposes_centroids_and_threshold():
    index = build_instance_index(_three_plane_nucleus(), threshold=0.4)

    assert index.threshold == 0.4
    np.testing.assert_allclose(index.centroid(0, 1), (3.5, 3.5))


def test_instance_index_threshold_changes_grouping():
    masks = _weakly_overlapping()

    merged = build_instance_index(masks, threshold=0.25)
    split = build_instance_index(masks, threshold=0.5)

    assert merged.plane_label_to_instance[(0, 1)] == (
        merged.plane_label_to_instance[(1, 1)]
    )
    assert split.plane_label_to_instance[(0, 1)] != (
        split.plane_label_to_instance[(1, 1)]
    )


def test_centroid_table_handles_sparse_label_values():
    masks = np.zeros((1, 10, 10), dtype=np.uint16)
    _block(masks, 0, 65535, 4, 8, 4, 8)

    table = centroid_table(masks)

    assert set(table) == {(0, 65535)}
    np.testing.assert_allclose(table[(0, 65535)], (5.5, 5.5))
