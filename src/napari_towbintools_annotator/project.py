class Project:
    def __init__(self, name: str, image_type: str, project_type: str, data_directories: list, project_dir: str, images_to_annotate: list):
        self.name = name
        self.image_type = image_type
        self.project_type = project_type
        self.data_directories = data_directories
        self.project_dir = project_dir
        self.images_to_annotate = images_to_annotate
        self.annotated_images = []
        self.ignored_images = []
        
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
