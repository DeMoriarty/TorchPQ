import torch
import cupy as cp
import numpy as np
import math
from custom_kernel import CustomKernel

class BMMCuda(CustomKernel): 
  def __init__(self, patch_m=4, patch_n=4):
    super(BMMCuda, self).__init__()
    self.patch_m = patch_m
    self.patch_n = patch_n
    
    with open("kernels/bmm_helpers.cu", "r") as f:
      helpers = f.read()

    with open("kernels/bmm.cu",'r') as f: ###
      self.kernel = helpers + f.read()
      
    self.kernel = (self.kernel
      .replace("_PM_", str(self.patch_m))
      .replace("_PN_", str(self.patch_n))
      .replace("__DISTANCE_FN__", "madd")
    )
    
    self._fn_tt = cp.RawKernel(
      code=self.kernel,
      name="bmm_tt",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )
    self._fn_nn = cp.RawKernel(
      code=self.kernel,
      name="bmm_nn",
      backend='nvcc',
      options=(
        '--maxrregcount=128',
        '--use_fast_math',
        #'-Xptxas',
        #'-dlcm=cg',
      )
    )
    # print(self._fn_nn.attributes)
    self._fn_tn = cp.RawKernel(
      code=self.kernel,
      name="bmm_tn",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )
    self._fn_nt = cp.RawKernel(
      code=self.kernel,
      name="bmm_nt",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )
  
  def get_mode(self, A, B):
    mode = [None, None]
    if A.stride()[-1] == 1:
      mode[0] = "n"
    elif A.stride()[-2] == 1:
      mode[0] = "t"
    if B.stride()[-1] == 1:
      mode[1] = "n"
    elif B.stride()[-2] == 1:
      mode[1] = "t"
    return "".join(mode)

  def __call__(self, A, B):
    """
      Performs C = f(A) @ g(B)
      A: torch.Tensor, shape : [l, m, k] or [l, k, m]
      B: torch.Tensor, shape : [l, n, k] or [l, k, n]
      returns C: torch.Tensor, shape : [l, m, n]
    """
    assert len(A.shape) == len(B.shape)
    # A = A.contiguous()
    # B = B.contiguous()
    if len(A.shape) == 2 and len(B.shape) == 2:
      A = A[None]
      B = B[None]
      two_dimentional = True
      dim += 1
    elif len(A.shape) == 3 and len(B.shape) == 3:
      two_dimentional = False
      pass
    else:
      raise ValueError("A and B need to be 2d or 3d")
    assert A.shape[0] == B.shape[0]
    assert A.shape[2] == B.shape[1]
    assert A.dtype == B.dtype
    assert A.dtype in [torch.float, torch.half]
    assert A.device.type == B.device.type == "cuda"

    mode = self.get_mode(A, B)

    if mode == "nn":
      kernel_fn = self._fn_nn
    elif mode == "tt":
      kernel_fn = self._fn_tt
    elif mode == "tn":
      kernel_fn = self._fn_tn
    elif mode == "nt":
      kernel_fn = self._fn_nt

    l, m, k = A.shape
    l, k, n = B.shape

    C = torch.zeros([l, m, n], device="cuda:0", dtype=A.dtype)

    threads_per_block = (256,)
    #blocks_per_grid = (math.ceil(n/128), math.ceil(m/128), l)
    
    n_ = math.ceil(n/(128*self.patch_n))
    m_ = math.ceil(m/(128*self.patch_m))
    blocks_per_grid = (self.patch_n*self.patch_m, n_ * m_, l)

    self._fn_nn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        m, n, k,
      ],
      stream=self.stream
    )

    if two_dimentional:
      C = C[0]
    return C