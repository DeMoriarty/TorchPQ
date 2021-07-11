from abc import ABC, abstractmethod

class BaseIndex(ABC):
  def __init__(self):
    super().__init__()
  
  @abstractmethod
  def add(self):
    pass

  @abstractmethod
  def remove(self):
    pass

  @abstractmethod
  def search(self):
    pass