import torch
from time import time
n_iter = 30
n_warmup_iter=3

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
a = torch.randint(256, size=[1000, 128], device="cuda:0")
b = torch.randint(256, size=[128, 100_000], device="cuda:0")




# tm = time()
# c1 = a.float() @ b.float()

def half_mm(a, b):
  a2 = a.float() / a.max()
  # b2 = b.float() / b.max()
  b2 = b.float()
  c2 = a2 @ b2
  # c2 = c2.float(.00) * (a.max() * b.max())
  return c2

def single_mm(a, b):
  a2 = a.float()
  b2 = b.float()
  c1 = a2 @ b2
  return c1

c1, rt1 = torch_timeit(single_mm, a, b)

c2, rt2 = torch_timeit(half_mm, a, b)
# print(c1)
# print(c2)
print(c1.topk(dim=-1, k=32)[1])
print(c2.topk(dim=-1, k=32)[1])
dif = (c1 - c2).pow(2)
print("MSE:", dif.mean())