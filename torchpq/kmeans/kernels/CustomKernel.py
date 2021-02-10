import cupy as cp
import torch

@cp.util.memoize(for_each_device=True)
def cunnex(func_name, func_body):
  return cp.cuda.compile_with_cache(func_body).get_function(func_name)
  # return cp.cuda.compile_with_cache(globals()[strFunction]).get_function(strFunction)

class Stream:
  def __init__(self, ptr):
    self.ptr = ptr
  
class CustomKernel:
  def __init__(self):
    self._use_torch_in_cupy_malloc()
    self.stream = Stream(torch.cuda.current_stream().cuda_stream)
    
  @staticmethod
  def _torch_alloc(size):
    device = cp.cuda.Device().id
    tensor = torch.empty(size, dtype=torch.uint8, device=device)
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