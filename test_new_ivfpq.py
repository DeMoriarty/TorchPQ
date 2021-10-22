import torch
from matplotlib import pyplot as plt
import numpy as np
from time import time

from torchpq.experimental.DistributedIVFPQIndex import DistributedIVFPQIndex
from torchpq.index import IVFPQIndex
from torchpq.fn import Topk
from torchpq import metric
from torchpq import util

d_vector = 128
n_subvectors = 8
n_cells = 100
initial_size = 128
expand_mode = "double"
pq_use_residual=False
verbose=1

# test init
index = DistributedIVFPQIndex(
  d_vector=d_vector,
  n_subvectors=n_subvectors,
  n_cells=n_cells,
  initial_size=initial_size,
  expand_mode=expand_mode,
  device="cuda:0",
  pq_use_residual=pq_use_residual,
  verbose=verbose
)