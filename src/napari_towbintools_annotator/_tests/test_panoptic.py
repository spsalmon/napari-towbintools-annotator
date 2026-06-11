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


from napari_towbintools_annotator.project_creator import scan_panoptic_files


def test_scan_panoptic_files_matches(tmp_path):
    ref = tmp_path / "ref"
    seg = tmp_path / "seg"
    ref.mkdir()
    seg.mkdir()
    for name in ("a.tif", "b.tif"):
        (ref / name).write_text("x")
        (seg / name).write_text("x")
    refs, segs = scan_panoptic_files([str(ref)], [str(seg)])
    assert len(refs) == 2
    assert len(segs) == 2
    assert all(r.endswith(".tif") for r in refs)


def test_scan_panoptic_files_mismatch_raises(tmp_path):
    ref = tmp_path / "ref"
    seg = tmp_path / "seg"
    ref.mkdir()
    seg.mkdir()
    (ref / "a.tif").write_text("x")
    (seg / "a.tif").write_text("x")
    (seg / "b.tif").write_text("x")
    with pytest.raises(ValueError):
        scan_panoptic_files([str(ref)], [str(seg)])


def test_run_panoptic_creation_copies_segmentations(tmp_path):
    from napari_towbintools_annotator.project_creator import ProjectCreatorWidget

    src_ref = tmp_path / "src_ref"
    src_seg = tmp_path / "src_seg"
    src_ref.mkdir()
    src_seg.mkdir()
    (src_ref / "a.tif").write_text("x")
    (src_seg / "a.tif").write_text("x")
    project_dir = tmp_path / "proj"

    class _Status:
        def emit(self, *args, **kwargs):
            pass

    # Call the unbound method without running QWidget.__init__.
    widget = ProjectCreatorWidget.__new__(ProjectCreatorWidget)
    ProjectCreatorWidget._run_panoptic_creation(
        widget,
        "proj",
        "multichannel",
        str(project_dir),
        [str(src_ref)],
        [str(src_seg)],
        ["a"],
        True,  # copy_data
        _Status(),
    )

    master = pd.read_csv(project_dir / "annotations" / "annotations.csv")
    assert len(master) == 1
    seg_path = str(master.loc[0, "Segmentation"])
    ref_path = str(master.loc[0, "Reference"])
    assert str(project_dir) in seg_path
    assert str(project_dir) in ref_path


import tifffile

from napari_towbintools_annotator.panoptic_annotator import (
    PanopticAnnotatorWidget,
)


def test_panoptic_widget_load_and_save(tmp_path):
    import napari

    project_dir = tmp_path / "proj"
    annotations_dir = project_dir / "annotations"
    annotations_dir.mkdir(parents=True)

    reference = np.zeros((10, 10), dtype=np.uint8)
    segmentation = np.zeros((10, 10), dtype=np.uint16)
    segmentation[2:4, 2:4] = 5
    ref_path = tmp_path / "img.tif"
    seg_path = tmp_path / "img_seg.tif"
    tifffile.imwrite(str(ref_path), reference)
    tifffile.imwrite(str(seg_path), segmentation)

    pd.DataFrame(
        {
            "Reference": [str(ref_path)],
            "Segmentation": [str(seg_path)],
            "Annotation": [""],
        }
    ).to_csv(annotations_dir / "annotations.csv", index=False)

    project = PanopticProject(
        name="p",
        image_type="multichannel",
        annotation_directories=["annotations"],
        annotation_df_path="annotations/annotations.csv",
        data_directories=[str(tmp_path)],
        mask_directories=[str(tmp_path)],
        classes=["a", "b"],
        project_dir=str(project_dir),
    )

    viewer = napari.Viewer(show=False)
    try:
        widget = PanopticAnnotatorWidget(viewer, project)
        assert widget.file_list_widget.count() == 1
        assert widget._annotation_layer is not None

        # Drop one point on instance 5, colored as class 0 ("a").
        widget._annotation_layer.data = np.array([[3, 3]])
        widget._annotation_layer.face_color = np.array(
            [widget.class_id_to_color[0]], dtype=float
        )
        widget.save_annotations()
        widget._save_master_sync()
    finally:
        viewer.close()

    out_csv = annotations_dir / "img.csv"
    assert out_csv.exists()
    saved = pd.read_csv(out_csv)
    assert int(saved.loc[0, "Label"]) == 5
    assert int(saved.loc[0, "ClassID"]) == 0
    assert saved.loc[0, "Class"] == "a"

    master = pd.read_csv(annotations_dir / "annotations.csv")
    assert str(master.loc[0, "Annotation"]) == str(out_csv)
