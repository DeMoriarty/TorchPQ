from os import path
import torch

def get_absolute_path(*relative_path):
  relative_path = path.join(*relative_path)
  return path.join(path.dirname(__file__), relative_path)

def str2dtype(dtype_str):
  if dtype_str in ["double", "float64"]:
    dtype = torch.float64
  elif dtype_str in ["half", "float16"]:
    dtype = torch.float16
  elif dtype_str in ["float", "float32"]:
    dtype = torch.float32
  elif dtype_str in ["bfloat16"]:
    dtype = torch.bfloat16
  elif dtype_str in ["long", "int64"]:
    dtype = torch.int64
  elif dtype_str in ["int", "int32"]:
    dtype = torch.int32
  elif dtype_str in ["int16"]:
    dtype = torch.int16
  elif dtype_str in ["int8"]:
    dtype = torch.int8
  elif dtype_str in ["uint8"]:
    dtype = torch.uint8
  elif dtype_str in ["bool"]:
    dtype = torch.bool
  else:
    raise TypeError(f"Unrecognized dtype string: {dtype_str}")
  return dtype

def check_device(tensor, *device):
  device = [torch.device(i) if type(i) == str else i for i in device]
  return tensor.device in device

def normalize(x, dim=0):
  """
    normalize given tensor along given dimention
  """
  x_norm = x.norm(dim=dim, keepdim=True) + 1e-9
  return x / x_norm

def get_compute_capability(device_id=0):
  """
    return compute capability of GPU device
  """
  if torch.cuda.is_available():
    # device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    result = (gpu_properties.major, gpu_properties.minor)
  else:
    result = (-1, 0)
  return result

def get_tensor_core_availability(device_id=0):
  cc = get_compute_capability(device_id)
  device_name = torch.cuda.get_device_name()
  if cc[0] > 7 and "GTX" not in device_name:
    return True
  return False

def get_maximum_shared_memory_bytes(device_id=0):
  cc = get_compute_capability(device_id)
  if cc[0] < 7 or cc == (7, 2):
    y = 48
  elif cc == (7, 0):
    y = 96
  elif cc == (7, 5):
    y = 64
  elif cc == (8, 0):
    y = 163
  elif cc == (8, 6):
    y = 99
  else:
    y = 0
  return y * 1024

def check_dtype(tensor, *dtype):
  dtype = [str2dtype(i) if type(i) == str else i for i in dtype]
  return tensor.dtype in dtype