from os import path
def get_absolute_path(relative_path):
  return path.join(path.dirname(__file__), relative_path)