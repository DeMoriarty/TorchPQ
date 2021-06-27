import torch
import numpy as np
from ..container import FlatContainer

class FlatIndex(FlatContainer):
  def __init__(
      self,
      d_vector,
      initial_size=None,
      expand_step_size=1024,
      expand_mode="double",
      device="cuda:0",
      distance=""
    ):
    super(FlatIndex, self).__init__()

  def train(self, x):
    pass
  
  def encode(self, x):
    pass

  def decode(self, x):
    pass
  
  def add(self, x, ids=None):
    pass

  def remove(self, ids=None, address=None):
    pass

  def search(self, queries, k=1):
    pass