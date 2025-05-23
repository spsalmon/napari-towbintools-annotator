from napari_towbintools_annotator.project import Project


def test_create_project():
    project = Project("Test Project")
    print(f"Project name: {project.name}")
