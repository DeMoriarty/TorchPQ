import cupy as cp
import torch
from torchpq.kernels.default_device import get_default_device

@cp.memoize(for_each_device=True)
def cunnex(func_name, func_body):
  return cp.cuda.compile_with_cache(func_body).get_function(func_name)

class Stream:
  def __init__(self, ptr):
    self.ptr = ptr
  
class CustomKernel:
  def __init__(self):
    self._use_torch_in_cupy_malloc()
    self.stream = Stream(torch.cuda.current_stream(get_default_device()).cuda_stream)

  @staticmethod
  def _torch_alloc(size):
    tensor = torch.empty(size, dtype=torch.uint8, device=get_default_device())
    return cp.cuda.MemoryPointer(
        cp.cuda.UnownedMemory(tensor.data_ptr(), size, tensor), 0)

  def _use_torch_in_cupy_malloc(self):
    cp.cuda.set_allocator(self._torch_alloc)

  def _compile_kernel_str(
      self,
      kernel,
      name,
      options=(),
      backend="nvrtc",
      max_dynamic_smem=None
    ):
    fn = cp.RawKernel(
      kernel,
      name,
      options=options,
      backend=backend,
    )
    if max_dynamic_smem:
      fn.max_dynamic_shared_size_bytes = max_dynamic_smem
    return fn