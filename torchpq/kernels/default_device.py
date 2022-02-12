import cupy as cp
import torch
__device = cp.cuda.Device().id

def get_default_device():
  global __device
  return __device

def set_default_device(device_id):
  assert device_id < torch.cuda.device_count()
  global __device
  __device = device_id