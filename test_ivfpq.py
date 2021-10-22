import torch
from matplotlib import pyplot as plt
import numpy as np
from time import time

from torchpq.index import IVFPQIndex
from torchpq.fn import Topk
from torchpq import metric
from torchpq import util
from torchpq.experimental.IVFPQIndex_v2 import IVFPQIndex_v2
from torchpq.experimental.IVFPQ4Index_v1 import IVFPQ4Index_v1

util.__silent = True

from test_util import *


n_data = 200_000
n_query = 1000
d_vector = 128
n_iter = 3
n_warmup_iter=1


warmup_matmul()
warmup_topk()

def setup_v1(d_vector, n_subvectors, n_cells, n_probe=1, verbose=0, **kwargs):
  index = IVFPQIndex(
    d_vector=d_vector,
    n_subvectors=n_subvectors,
    n_cells=n_cells,
    verbose=verbose,
    initial_size=2048,
    **kwargs,
  )

  index.train(data)
  index.add(data)

  index.n_probe = n_probe
  index.use_smart_probing = True
  if "pq_use_residual" in kwargs.keys() and kwargs["pq_use_residual"]:
    index.use_precomputed = True
  if index.use_smart_probing:
    index.smart_probing_temperature = 10
  return index

def setup_v2(d_vector, n_subvectors, n_cells, n_probe=1, verbose=0, **kwargs):
  index = IVFPQIndex_v2(
    d_vector=d_vector,
    n_subvectors=n_subvectors,
    n_cells=n_cells,
    verbose=verbose,
    initial_size=2048,
    **kwargs,
  )

  index.train(data)
  index.add(data)

  index.n_probe = n_probe
  index.use_smart_probing = True
  if "pq_use_residual" in kwargs.keys() and kwargs["pq_use_residual"]:
    index.use_precomputed = True
  if index.use_smart_probing:
    index.smart_probing_temperature = 10
  return index

def setup_v3(d_vector, n_subvectors, n_cells, n_probe=1, verbose=0, **kwargs):
  index = IVFPQ4Index_v1(
    d_vector=d_vector,
    n_subvectors=n_subvectors,
    n_cells=n_cells,
    verbose=verbose,
    initial_size=2048*2,
    **kwargs,
  )

  index.train(data)
  index.add(data)

  index.n_probe = n_probe
  index.use_smart_probing = True
  if "pq_use_residual" in kwargs.keys() and kwargs["pq_use_residual"]:
    index.use_precomputed = True
  if index.use_smart_probing:
    index.smart_probing_temperature = 10
  return index


n_subvectors=32
n_cells=512
n_probe = 16
k = 1
pq_use_residual=False
assert k == 1

data = torch.randn(
  d_vector,
  n_data,
  device="cuda:0"
)

query = torch.randn(
  d_vector,
  n_query,
  device="cuda:0"
)

sims = metric.negative_squared_l2_distance(query, data)
_, groundtruth = sims.topk(k=k, dim=-1)

# ivfpq_topk_v1
index = setup_v1(d_vector, n_subvectors, n_cells, n_probe, pq_use_residual=pq_use_residual)
(topkv_v1, topki_v1), rt1 = torch_timeit(
  index.search, 
  query,
  k=k,
  n_iter=n_iter, 
  n_warmup_iter=n_warmup_iter, 
)
print(f"recall@{k} v1:", recall(topki_v1, groundtruth, k))
print(f"overlap@{k} v1:", overlap(topki_v1, groundtruth))

# ivfpq_topk_v1
index = setup_v2(d_vector, n_subvectors, n_cells, n_probe, pq_use_residual=pq_use_residual)
(topkv_v2, topki_v2), rt2 = torch_timeit(
  index.search, 
  query,
  k=k,
  n_iter=n_iter, 
  n_warmup_iter=n_warmup_iter, 
)
print(topki_v2.shape, groundtruth.shape)
print(f"recall@{k} v2:", recall(topki_v2, groundtruth, k))
print(f"overlap@{k} v2:", overlap(topki_v2, groundtruth))

# ivfpq4_topk_v1
index = setup_v3(d_vector, n_subvectors * 2, n_cells, n_probe, pq_use_residual=pq_use_residual)
(topkv_v3, topki_v3), rt3 = torch_timeit(
  index.search, 
  query,
  k=k,
  n_iter=n_iter, 
  n_warmup_iter=n_warmup_iter, 
)
print(topki_v3.shape, groundtruth.shape)
print(f"recall@{k} v3:", recall(topki_v3, groundtruth, k))
print(f"overlap@{k} v3:", overlap(topki_v3, groundtruth))
