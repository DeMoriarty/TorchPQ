import torch
from abc import ABC, abstractmethod
from ..CustomModule import CustomModule

class BaseIndex(CustomModule, ABC):
  def __init__():
    super(BaseIndex, self).__init__()
  
  @abstractmethod
  def add(self):
    pass

  @abstractmethod
  def remove(self):
    pass

  @abstractmethod
  def search(self):
    pass