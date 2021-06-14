import torch
import numpy as np
from .BaseIndex import BaseIndex

class IVFSQIndex(BaseIndex):
  def __init__(self):
    super(IVFSQIndex, self).__init__()

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