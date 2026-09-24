import os

import numpy as np
import tifffile
from napari_align_annotator.stitching import build_instance_index
from napari_align_annotator.stitching import get_instance_index
from napari_align_annotator.stitching import load_cached_index
from napari_align_annotator.stitching import save_index_to_cache


def _stack():
    masks = np.zeros((3, 20, 20), dtype=np.uint16)
    masks[0, 2:6, 2:6] = 1
    masks[1, 2:6, 2:6] = 7
    masks[2, 2:6, 2:6] = 3
    masks[1, 14:18, 14:18] = 8
    return masks


def _write_stack(tmp_path, masks=None):
    path = tmp_path / "seg.tif"
    tifffile.imwrite(str(path), _stack() if masks is None else masks)
    return str(path)


def test_cache_round_trips_an_index(tmp_path):
    masks = _stack()
    seg = _write_stack(tmp_path)
    cache = tmp_path / "cache"
    index = build_instance_index(masks, threshold=0.25)

    save_index_to_cache(str(cache), seg, index)
    loaded = load_cached_index(str(cache), seg, 0.25, masks)

    assert loaded is not None
    assert loaded.plane_label_to_instance == index.plane_label_to_instance
    assert loaded.instance_to_planes == index.instance_to_planes
    assert loaded.threshold == index.threshold
    for key, value in index.centroids.items():
        np.testing.assert_allclose(loaded.centroids[key], value)


def test_cache_misses_when_threshold_differs(tmp_path):
    masks = _stack()
    seg = _write_stack(tmp_path)
    cache = tmp_path / "cache"
    save_index_to_cache(
        str(cache), seg, build_instance_index(masks, threshold=0.25)
    )

    assert load_cached_index(str(cache), seg, 0.5, masks) is None


def test_cache_misses_when_segmentation_changes(tmp_path):
    masks = _stack()
    seg = _write_stack(tmp_path)
    cache = tmp_path / "cache"
    save_index_to_cache(
        str(cache), seg, build_instance_index(masks, threshold=0.25)
    )
    assert load_cached_index(str(cache), seg, 0.25, masks) is not None

    # Re-segmenting the same path must not serve a stale index.
    changed = _stack()
    changed[2, 2:6, 2:6] = 0
    tifffile.imwrite(seg, changed)
    os.utime(seg, ns=(2_000_000_000_000_000_000, 2_000_000_000_000_000_000))

    assert load_cached_index(str(cache), seg, 0.25, changed) is None


def test_cache_treats_corrupt_file_as_a_miss(tmp_path):
    masks = _stack()
    seg = _write_stack(tmp_path)
    cache = tmp_path / "cache"
    save_index_to_cache(
        str(cache), seg, build_instance_index(masks, threshold=0.25)
    )

    for name in os.listdir(str(cache)):
        (cache / name).write_bytes(b"not an npz file")

    assert load_cached_index(str(cache), seg, 0.25, masks) is None


def test_cache_save_is_silent_when_directory_is_unwritable(tmp_path):
    masks = _stack()
    seg = _write_stack(tmp_path)
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    os.chmod(str(blocked), 0o500)
    try:
        # Must not raise; caching is an optimisation, never a hard failure.
        save_index_to_cache(
            str(blocked / "cache"),
            seg,
            build_instance_index(masks, threshold=0.25),
        )
    finally:
        os.chmod(str(blocked), 0o700)


def test_get_instance_index_builds_then_serves_from_cache(tmp_path):
    masks = _stack()
    seg = _write_stack(tmp_path)
    cache = tmp_path / "cache"

    first = get_instance_index(seg, masks, 0.25, str(cache))
    assert first.plane_label_to_instance

    # Second call must come from disk: corrupt the source array so a rebuild
    # would produce a different answer, then check we still get the original.
    second = get_instance_index(seg, np.zeros_like(masks), 0.25, str(cache))
    assert second.plane_label_to_instance == first.plane_label_to_instance


def test_get_instance_index_works_without_a_cache_dir(tmp_path):
    masks = _stack()
    seg = _write_stack(tmp_path)

    index = get_instance_index(seg, masks, 0.25, None)

    assert index.plane_label_to_instance[(0, 1)] == (
        index.plane_label_to_instance[(1, 7)]
    )
