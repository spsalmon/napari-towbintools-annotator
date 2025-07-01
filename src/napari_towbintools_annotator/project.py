import yaml

class Project:
    def __init__(self, name: str, image_type: str, project_type: str, annotation_directories: list,  data_directories: list, project_dir: str, classes: list = [], ignored_images: list = []):
        self.name = name
        self.image_type = image_type
        self.project_type = project_type
        self.annotation_directories = annotation_directories
        self.data_directories = data_directories
        self.project_dir = project_dir

        if project_type == "classification" and not classes:
            raise ValueError("Classes must be provided for classification projects.")
        
        self.classes = classes
        self.ignored_images = ignored_images
        
    def __str__(self):
        return f"Project(name={self.name}, image_type={self.image_type}, project_type={self.project_type}, data_directories={self.data_directories}, project_dir={self.project_dir})"

    def add_data(self, key: str, value):
        self.data[key] = value

    def add_annotation(self, key: str, value):
        self.annotations[key] = value

    def get_data(self, key: str):
        return self.data.get(key)

    def get_annotation(self, key: str):
        return self.annotations.get(key)
    
    def save(self):
        project_data = {
            'name': self.name,
            'image_type': self.image_type,
            'project_type': self.project_type,
            'annotation_directories': self.annotation_directories,
            'data_directories': self.data_directories,
            'project_dir': self.project_dir,
            'ignored_images': self.ignored_images,
            'classes': self.classes,
        }
        
        with open(f"{self.project_dir}/project.yaml", 'w') as file:
            yaml.dump(project_data, file)

    @classmethod
    def load(cls, project_dir: str):
        with open(f"{project_dir}/project.yaml", 'r') as file:
            project_data = yaml.safe_load(file)

        project_type = project_data.get('project_type', None)
        if project_type is None:
            raise ValueError("Project type is not specified in the project file.")
        
        if project_type == "classification":
            return cls(
                name=project_data['name'],
                image_type=project_data['image_type'],
                project_type=project_data['project_type'],
                annotation_directories=project_data['annotation_directories'],
                data_directories=project_data['data_directories'],
                project_dir=project_dir,
                classes=project_data.get('classes', []),
                ignored_images=project_data.get('ignored_images', [])
            )