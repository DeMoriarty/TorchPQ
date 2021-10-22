import torch
from time import time, sleep

a = torch.randn(512, 512, device="cpu")
b = torch.randn(512, 512, device="cpu")

tm = time()
c = a @ b
# torch.cuda.synchronize()
print(c[0,0])
print("time spent for CPU matmul:", time() - tm)

a = a.cuda(0)
b = b.cuda(0)

c = a @ b
torch.cuda.synchronize()
sleep(60)


tm = time()
for i in range(1):
  c = a @ b
  # if i == 0:
  #   tm = time()
# print(c[0,0])
torch.cuda.synchronize()
print("time spent for GPU matmul:", (time() - tm) / 1)



# import cupy
# a = cupy.random.randn(512, 512, dtype=cupy.float32)
# b = cupy.random.randn(512, 512, dtype=cupy.float32)

# print(a.shape)
# print(b.shape)
# cupy.cuda.Stream.null.synchronize()
# tm = time()
# c = cupy.matmul(a, b)
# print(c.shape)
# # torch.cuda.synchronize()
# cupy.cuda.Stream.null.synchronize()
# print("time spent for CuPy matmul:", time() - tm)