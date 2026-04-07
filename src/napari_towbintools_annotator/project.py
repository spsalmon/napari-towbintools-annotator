import yaml

class Project:
    def __init__(
        self,
        name: str,
        image_type: str,
        project_type: str,
        annotation_directories: list,
        data_directories: list,
        project_dir: str,
        ignored_images: list = None,
    ):
        self.name = name
        self.image_type = image_type
        self.project_type = project_type
        self.annotation_directories = annotation_directories
        self.data_directories = data_directories
        self.project_dir = project_dir
        self.ignored_images = ignored_images

    def __str__(self):
        return (
            f"Project(name={self.name}, image_type={self.image_type}, "
            f"project_type={self.project_type}, data_directories={self.data_directories}, "
            f"project_dir={self.project_dir})"
        )

    def save(self):
        raise NotImplementedError("Subclasses must implement save().")

    @classmethod
    def load(cls, project_dir: str):
        with open(f"{project_dir}/project.yaml") as file:
            project_data = yaml.safe_load(file)

        project_type = project_data.get("project_type")
        if project_type is None:
            raise ValueError("Project type is not specified in the project file.")

        dispatch = {
            "classification": ClassificationProject.load,
        }

        if project_type not in dispatch:
            raise NotImplementedError(f"Loading project type '{project_type}' is not yet supported.")

        return dispatch[project_type](project_dir, project_data)


class ClassificationProject(Project):
    def __init__(
        self,
        name: str,
        image_type: str,
        annotation_directories: list,
        annotation_df_path: str,
        data_directories: list,
        project_dir: str,
        classes: list,
        ignored_images: list = None,
    ):
        if not classes:
            raise ValueError("Classes must be provided for classification projects.")

        super().__init__(
            name=name,
            image_type=image_type,
            project_type="classification",
            annotation_directories=annotation_directories,
            data_directories=data_directories,
            project_dir=project_dir,
            ignored_images=ignored_images,
        )

        self.annotation_df_path = annotation_df_path
        self.classes = classes

    def save(self):
        project_data = {
            "name": self.name,
            "image_type": self.image_type,
            "project_type": self.project_type,
            "annotation_directories": self.annotation_directories,
            "annotation_df_path": self.annotation_df_path,
            "data_directories": self.data_directories,
            "project_dir": self.project_dir,
            "ignored_images": self.ignored_images,
            "classes": self.classes,
        }

        with open(f"{self.project_dir}/project.yaml", "w") as file:
            yaml.dump(project_data, file)

    @classmethod
    def load(cls, project_dir: str, project_data: dict):
        return cls(
            name=project_data["name"],
            image_type=project_data["image_type"],
            annotation_directories=project_data["annotation_directories"],
            annotation_df_path=project_data["annotation_df_path"],
            data_directories=project_data["data_directories"],
            project_dir=project_dir,
            classes=project_data.get("classes", []),
            ignored_images=project_data.get("ignored_images", []),
        )