from napari_towbintools_annotator.colors import (
    CLASS_PALETTE,
    class_hex,
    hex_to_rgba_float,
)


def test_hex_to_rgba_float_white():
    assert hex_to_rgba_float("#ffffff") == (1.0, 1.0, 1.0, 1.0)


def test_hex_to_rgba_float_strips_hash_and_scales():
    r, g, b, a = hex_to_rgba_float("#4C72B0")
    assert a == 1.0
    assert abs(r - 0x4C / 255) < 1e-9
    assert abs(g - 0x72 / 255) < 1e-9
    assert abs(b - 0xB0 / 255) < 1e-9


def test_class_hex_indexes_palette_and_wraps():
    classes = list(range(len(CLASS_PALETTE) + 2))
    assert class_hex(classes, 0) == CLASS_PALETTE[0]
    assert class_hex(classes, len(CLASS_PALETTE)) == CLASS_PALETTE[0]


import pytest

from napari_towbintools_annotator.project import PanopticProject, Project


def _make_panoptic_project(tmp_path, classes=("a", "b")):
    return PanopticProject(
        name="p",
        image_type="multichannel",
        annotation_directories=["annotations"],
        annotation_df_path="annotations/annotations.csv",
        data_directories=[str(tmp_path / "ref")],
        mask_directories=[str(tmp_path / "seg")],
        classes=list(classes),
        project_dir=str(tmp_path),
    )


def test_panoptic_project_save_load_roundtrip(tmp_path):
    proj = _make_panoptic_project(tmp_path)
    proj.save()
    loaded = Project.load(str(tmp_path))
    assert isinstance(loaded, PanopticProject)
    assert loaded.project_type == "panoptic"
    assert loaded.classes == ["a", "b"]
    assert loaded.mask_directories == [str(tmp_path / "seg")]
    assert loaded.data_directories == [str(tmp_path / "ref")]
    assert loaded.annotation_df_path == "annotations/annotations.csv"


def test_panoptic_project_requires_classes(tmp_path):
    with pytest.raises(ValueError):
        PanopticProject(
            name="p",
            image_type="multichannel",
            annotation_directories=["annotations"],
            annotation_df_path="annotations/annotations.csv",
            data_directories=[str(tmp_path / "ref")],
            mask_directories=[str(tmp_path / "seg")],
            classes=[],
            project_dir=str(tmp_path),
        )


def test_panoptic_project_requires_mask_dirs(tmp_path):
    with pytest.raises(ValueError):
        PanopticProject(
            name="p",
            image_type="multichannel",
            annotation_directories=["annotations"],
            annotation_df_path="annotations/annotations.csv",
            data_directories=[str(tmp_path / "ref")],
            mask_directories=[],
            classes=["a"],
            project_dir=str(tmp_path),
        )


import numpy as np
import pandas as pd

from napari_towbintools_annotator.panoptic_annotator import (
    nearest_class_id,
    points_to_rows,
    rows_to_points,
)


def test_nearest_class_id_handles_float_noise():
    id_to_color = {0: (1, 0, 0, 1), 1: (0, 0, 1, 1)}
    assert nearest_class_id((0.98, 0.01, 0.0, 1.0), id_to_color) == 0
    assert nearest_class_id((0.0, 0.02, 0.97, 1.0), id_to_color) == 1


def test_points_to_rows_2d():
    label_data = np.zeros((10, 10), dtype=int)
    label_data[2:4, 2:4] = 5
    label_data[6:8, 6:8] = 9
    id_to_color = {0: (1, 0, 0, 1), 1: (0, 0, 1, 1)}
    id_to_name = {0: "a", 1: "b"}
    points = np.array([[3, 3], [7, 7]])
    colors = np.array([(1, 0, 0, 1), (0, 0, 1, 1)], dtype=float)
    rows = points_to_rows(points, colors, label_data, id_to_color, id_to_name)
    assert rows == [
        {"Label": 5, "ClassID": 0, "Class": "a"},
        {"Label": 9, "ClassID": 1, "Class": "b"},
    ]


def test_points_to_rows_3d_includes_plane():
    label_data = np.zeros((3, 10, 10), dtype=int)
    label_data[1, 2:4, 2:4] = 7
    id_to_color = {0: (1, 0, 0, 1)}
    id_to_name = {0: "a"}
    points = np.array([[1, 3, 3]])
    colors = np.array([(1, 0, 0, 1)], dtype=float)
    rows = points_to_rows(
        points, colors, label_data, id_to_color, id_to_name, plane_axis="Z"
    )
    assert rows == [{"Z": 1, "Label": 7, "ClassID": 0, "Class": "a"}]


def test_points_to_rows_skips_out_of_bounds():
    label_data = np.zeros((5, 5), dtype=int)
    rows = points_to_rows(
        np.array([[100, 100]]),
        np.array([(1, 0, 0, 1)], dtype=float),
        label_data,
        {0: (1, 0, 0, 1)},
        {0: "a"},
    )
    assert rows == []


def test_rows_to_points_2d_centroid():
    label_data = np.zeros((10, 10), dtype=int)
    label_data[2:4, 2:4] = 5
    df = pd.DataFrame([{"Label": 5, "ClassID": 0, "Class": "a"}])
    placements = rows_to_points(df, label_data, {0: (1, 0, 0, 1)})
    assert len(placements) == 1
    point, color = placements[0]
    assert color == (1, 0, 0, 1)
    np.testing.assert_allclose(point, [2.5, 2.5])


def test_rows_to_points_3d_centroid():
    label_data = np.zeros((3, 10, 10), dtype=int)
    label_data[1, 2:4, 2:4] = 7
    df = pd.DataFrame([{"Z": 1, "Label": 7, "ClassID": 0, "Class": "a"}])
    placements = rows_to_points(
        df, label_data, {0: (1, 0, 0, 1)}, plane_axis="Z"
    )
    assert len(placements) == 1
    point, _ = placements[0]
    np.testing.assert_allclose(point, [1, 2.5, 2.5])


def test_rows_to_points_skips_missing_label():
    label_data = np.zeros((10, 10), dtype=int)
    df = pd.DataFrame([{"Label": 99, "ClassID": 0, "Class": "a"}])
    assert rows_to_points(df, label_data, {0: (1, 0, 0, 1)}) == []


def test_points_to_rows_skips_background_label():
    label_data = np.zeros((10, 10), dtype=int)
    label_data[2:4, 2:4] = 5
    # First point lands on background (label 0), second on instance 5.
    points = np.array([[8, 8], [3, 3]])
    colors = np.array([(1, 0, 0, 1), (1, 0, 0, 1)], dtype=float)
    rows = points_to_rows(
        points, colors, label_data, {0: (1, 0, 0, 1)}, {0: "a"}
    )
    assert rows == [{"Label": 5, "ClassID": 0, "Class": "a"}]
