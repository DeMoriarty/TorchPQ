from os import path

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

def check_dtype(tensor, *dtype):
  dtype = [util._str2dtype(i) if type(i) == str else i for i in dtype]
  return tensor.dtype in dtype