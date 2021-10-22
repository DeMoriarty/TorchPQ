import torch
from torchpq.fn import Topk
from time import time

def warmup_matmul():
  a = torch.randn(128, 128, device="cuda:0")
  b = torch.randn(128, 128, device="cuda:0")
  c = a @ b

def warmup_topk():
  topk = Topk()
  a = torch.randn(10, 1000, device="cuda:0")
  topk(a, k=1)

def torch_timeit(func, *args, **kwargs):
  n_iter = kwargs["n_iter"] if "n_iter" in kwargs.keys() else 5
  n_warmup_iter = kwargs["n_warmup_iter"] if "n_warmup_iter" in kwargs.keys() else 1
  del kwargs["n_iter"]
  del kwargs["n_warmup_iter"]

  tm = time()
  for i in range(n_iter + n_warmup_iter):
    result = func(*args, **kwargs)
    torch.cuda.synchronize()
    if i == n_warmup_iter - 1:
      tm = time()
  rt = (time() - tm) / n_iter
  print(f"time spent for running {func.__qualname__}: {rt}")
  return result, rt

def recall(topki, groundtruth, k):
  r =  (groundtruth[:, 0][:, None] == topki[:, :k]).sum(dim=-1).float().mean()
  return r

def overlap(topki, groundtruth):
  o = (groundtruth[:, :, None] == topki[:, None, :]).sum(dim=-1).float().mean()
  return o
  