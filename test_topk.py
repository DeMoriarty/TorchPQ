import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from time import time

from torchpq.kernels.TopkSelectCuda import TopkSelectCuda
from torchpq.kernels.TopkSelectCuda_v2 import TopkSelectCuda_v2
from torchpq.kernels.TopkSelectCuda_v3 import TopkSelectCuda_v3
from torchpq.kernels.TopkSelectCuda_v4 import TopkSelectCuda_v4

def torch_timeit(func, *args, **kwargs):
  tm = time()
  for i in range(n_iter + n_warmup_iter):
    result = func(*args, **kwargs)
    torch.cuda.synchronize()
    if i == n_warmup_iter - 1:
      tm = time()
  rt = (time() - tm) / n_iter
  print(f"time spent for running {func.__qualname__}: {rt}")
  return result, rt
  

batch_size = 1000
n_data = 500_000
k = 32
n_warmup_iter = 3
n_iter = 25

x = torch.randn(batch_size, n_data, device="cuda:0")

# baseline: torch.topk 
(topkv_base, topki_base), rt_base = torch_timeit(x.topk, k=k, dim=-1)

# TopkSelectCuda v1
topk_fn_v1 = TopkSelectCuda(tpb = 32, queue_capacity=4, buffer_size=4)
(topkv_v1, topki_v1), rt_v1 = torch_timeit(topk_fn_v1.__call__, x, k=k, dim=1)

# TopkSelectCuda v2
topk_fn_v2 = TopkSelectCuda_v2(tpb = 256, queue_capacity=1, buffer_size=4)
(topkv_v2, topki_v2), rt_v2 = torch_timeit(topk_fn_v2.__call__, x, k=k, dim=1)

# TopkSelectCuda v3
x = x.half()
topk_fn_v3 = TopkSelectCuda_v3(tpb = 256, queue_capacity=1, buffer_size=4)
(topkv_v3, topki_v3), rt_v3 = torch_timeit(topk_fn_v3.__call__, x, k=k, dim=1)

# TopkSelectCuda v4
if k == 1:
  topk_fn_v4 = TopkSelectCuda_v4(tpb = 256, queue_capacity=1, buffer_size=4)
  (topkv_v4, topki_v4), rt_v4 = torch_timeit(topk_fn_v4.__call__, x, k=k, dim=1)

vdif = (topkv_base - topkv_v3).abs()
idif = (topki_base != topki_v3)
idx_union = topki_base[:, :, None] == topki_v3[:, None, :]
print("inclusion:", idx_union.sum(-1).float().mean())
print("value error", vdif.sum())
print("index error", idif.sum())