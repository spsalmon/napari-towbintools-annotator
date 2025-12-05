from napari_towbintools_annotator.project import Project


def test_create_project():
    _project = Project(
        name="test_project",
        image_type="multichannel",
        project_type="classification",
        annotation_directories=["./test_project/annotations"],
        data_directories="./test_images",
        classes=["worm", "egg", "error"],
        project_dir="./test_project",
    )
