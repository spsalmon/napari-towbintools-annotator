import pytest
import yaml
from napari_align_annotator.project import ClassificationProject
from napari_align_annotator.project import PanopticProject
from napari_align_annotator.project import Project


def _classification(tmp_path, **overrides):
    kwargs = {
        "name": "c",
        "image_type": "multichannel",
        "annotation_directories": ["annotations"],
        "annotation_df_path": "annotations/annotations.csv",
        "project_dir": str(tmp_path),
        "classes": ["alive", "dead"],
        "data_directories": ["/data/images"],
    }
    kwargs.update(overrides)
    return ClassificationProject(**kwargs)


def test_base_project_holds_its_fields():
    project = Project(
        name="test_project",
        image_type="multichannel",
        project_type="keypoint",
        annotation_directories=["annotations"],
        data_directories=["images"],
        project_dir="proj",
    )

    assert project.project_type == "keypoint"
    assert project.ignored_images is None
    text = str(project)
    assert "name=test_project" in text
    assert "project_type=keypoint" in text


def test_base_project_cannot_be_saved(tmp_path):
    project = Project("p", "zstack", "keypoint", [], [], str(tmp_path))

    with pytest.raises(NotImplementedError):
        project.save()


def test_classification_project_round_trips(tmp_path):
    _classification(
        tmp_path,
        display_mode="both",
        mask_directories=["/data/masks"],
        ignored_images=["x.tif"],
    ).save()

    loaded = Project.load(str(tmp_path))

    assert isinstance(loaded, ClassificationProject)
    assert loaded.project_type == "classification"
    assert loaded.classes == ["alive", "dead"]
    assert loaded.display_mode == "both"
    assert loaded.data_directories == ["/data/images"]
    assert loaded.mask_directories == ["/data/masks"]
    assert loaded.ignored_images == ["x.tif"]
    assert loaded.annotation_df_path == "annotations/annotations.csv"


def test_loading_uses_the_directory_it_was_loaded_from(tmp_path):
    """A project directory that was moved still loads from its new place."""
    project = _classification(tmp_path)
    project.save()
    data = yaml.safe_load((tmp_path / "project.yaml").read_text())
    data["project_dir"] = "/old/location"
    (tmp_path / "project.yaml").write_text(yaml.dump(data))

    assert Project.load(str(tmp_path)).project_dir == str(tmp_path)


def test_classification_optional_fields_default_on_load(tmp_path):
    (tmp_path / "project.yaml").write_text(
        yaml.dump(
            {
                "name": "old",
                "image_type": "multichannel",
                "project_type": "classification",
                "annotation_directories": ["annotations"],
                "annotation_df_path": "annotations/annotations.csv",
                "data_directories": ["/data"],
                "classes": ["a"],
            }
        )
    )

    loaded = Project.load(str(tmp_path))

    assert loaded.display_mode == "image"
    assert loaded.mask_directories == []
    assert loaded.ignored_images == []


def test_mask_only_classification_needs_no_image_directories(tmp_path):
    project = _classification(
        tmp_path,
        data_directories=None,
        display_mode="mask",
        mask_directories=["/data/masks"],
    )

    assert project.data_directories == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"classes": []}, "Classes"),
        ({"display_mode": "video"}, "display_mode"),
        ({"display_mode": "mask"}, "mask_directories"),
        ({"display_mode": "both"}, "mask_directories"),
        ({"data_directories": None}, "data_directories"),
    ],
)
def test_classification_project_rejects_bad_configuration(
    tmp_path, overrides, message
):
    with pytest.raises(ValueError, match=message):
        _classification(tmp_path, **overrides)


def test_panoptic_project_requires_data_directories(tmp_path):
    with pytest.raises(ValueError, match="data_directories"):
        PanopticProject(
            name="p",
            image_type="zstack",
            annotation_directories=["annotations"],
            annotation_df_path="annotations/annotations.csv",
            project_dir=str(tmp_path),
            classes=["a"],
            mask_directories=["/masks"],
        )


def test_load_rejects_a_file_without_project_type(tmp_path):
    (tmp_path / "project.yaml").write_text(yaml.dump({"name": "x"}))

    with pytest.raises(ValueError, match="Project type"):
        Project.load(str(tmp_path))


def test_load_rejects_an_unknown_project_type(tmp_path):
    (tmp_path / "project.yaml").write_text(
        yaml.dump({"name": "x", "project_type": "keypoint"})
    )

    with pytest.raises(NotImplementedError, match="keypoint"):
        Project.load(str(tmp_path))


def test_load_of_a_missing_project_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Project.load(str(tmp_path / "nope"))
