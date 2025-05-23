class Project:
    def __init__(self, name: str):
        self.name = name
        self.data = {}
        self.annotations = {}

    def add_data(self, key: str, value):
        self.data[key] = value

    def add_annotation(self, key: str, value):
        self.annotations[key] = value

    def get_data(self, key: str):
        return self.data.get(key)

    def get_annotation(self, key: str):
        return self.annotations.get(key)
