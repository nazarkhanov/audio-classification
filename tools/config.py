import yaml


class Config:
    @staticmethod
    def load(path):
        with open(path, 'r') as stream:
            return yaml.safe_load(stream)
