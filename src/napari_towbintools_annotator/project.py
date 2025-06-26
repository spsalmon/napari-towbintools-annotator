class Project:
    def __init__(self, name: str, image_type: str = 'multichannel'):
        self.name = name
        self.data = {}
        self.annotations = {}
        self.image_type = image_type

    def add_data(self, key: str, value):
        self.data[key] = value

    def add_annotation(self, key: str, value):
        self.annotations[key] = value

    def get_data(self, key: str):
        return self.data.get(key)

    def get_annotation(self, key: str):
        return self.annotations.get(key)
