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
