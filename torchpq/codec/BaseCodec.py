import torch
from ..CustomModule import CustomModule
from abc import ABC, abstractmethod

class BaseCodec(CustomModule, ABC):
  def __init__(self):
    super(BaseCodec, self).__init__()
    self.register_buffer("_is_trained", torch.tensor(False))

  def _trained(self, value):
    assert type(value) == bool
    self._is_trained.data = torch.tensor(value)

  @property
  def is_trained(self):
    return self._is_trained.item()

  @abstractmethod
  def train(self):
    pass

  @abstractmethod
  def encode(self):
    pass

  @abstractmethod
  def decode(self):
    pass
    