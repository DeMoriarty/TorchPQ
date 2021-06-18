import torch
import cupy as cp
import numpy as np
import math
from .CustomKernel import CustomKernel
from ..util import get_absolute_path

class TopkBMMCuda(CustomKernel): 
  def __init__(
      self, patch_m=4, patch_n=4,
      distance="inner"
    ):
    super(TopkBMMCuda, self).__init__()
    self.patch_m = patch_m
    self.patch_n = patch_n
    if distance == "inner":
      dist_fn = "madd"
    elif distance in ["l2", "euclidean"]:
      dist_fn = "squared_l2"
    elif distance in ["l1", "manhattan"]:
      dist_fn = "l1"
    else:
      ValueError("Unrecognized distance type")

    self.distance = distance

   with open(get_absolute_path("kernels", "cuda", "bmm_helpers.cu"), "r") as f:
      helpers = f.read()
    
    with open(get_absolute_path("kernels", "cuda", "topkbmm.cu"),'r') as f: ###
      self.kernel = helpers + f.read()
      
    self.kernel = (self.kernel
      .replace("_PM_", str(self.patch_m))
      .replace("_PN_", str(self.patch_n))
      .replace("__DISTANCE_FN__", dist_fn)
    )
    
    self._fn_tt = cp.RawKernel(
      code=self.kernel,
      name="topk_bmm_tt",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )
    self._fn_nn = cp.RawKernel(
      code=self.kernel,
      name="topk_bmm_nn",
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
      name="topk_bmm_tn",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )
    self._fn_nt = cp.RawKernel(
      code=self.kernel,
      name="topk_bmm_nt",
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

  def __call__(self, A, B, k=128, dim=1):
    """
      Performs C = min(f(A) @ g(B)), argmin(f(A) @ g(B))
      A: torch.Tensor, shape : [l, m, k]
      B: torch.Tensor, shape : [l, k, n]
      returns C: torch.Tensor, shape : [l, m, n]
    """
    assert len(A.shape) == len(B.shape)
    if len(A.shape) == 2 and len(B.shape) == 2:
      A = A[None]
      B = B[None]
      two_dimentional = True
      dim += 1
    elif len(A.shape) == 3 and len(B.shape) == 3:
      two_dimentional = False
    else:
      raise ValueError("shape of A and B need to be 2d or 3d")
    assert A.shape[0] == B.shape[0]
    assert A.shape[2] == B.shape[1]
    assert A.dtype == B.dtype
    assert A.dtype in [torch.float, torch.half]
    assert A.device.type == B.device.type == "cuda"
    assert dim in [1, 2]
    assert 0 < k <= 128

    mode = self.get_mode(A, B)
    if mode == "nn":
      kernel_fn = self._fn_nn
    elif mode == "nt":
      kernel_fn = self._fn_nt
    elif mode == "tn":
      kernel_fn = self._fn_tn
    elif mode == "tt":
      kernel_fn = self._fn_tt

    l, m, d = A.shape
    l, d, n = B.shape

    if dim == 1:
      values = torch.empty([l, n, 128], device="cuda:0", dtype=A.dtype)
      indices = torch.empty([l, n, 128], device="cuda:0", dtype=torch.int64)
      mutex = torch.zeros([l, n], device="cuda:0", dtype=torch.int32)
    elif dim == 2:
      values = torch.empty([l, m, 128], device="cuda:0", dtype=A.dtype)
      indices = torch.empty([l, m, 128], device="cuda:0", dtype=torch.int64)
      mutex = torch.zeros([l, m], device="cuda:0", dtype=torch.int32)
    values.fill_(float("-inf"))

    threads_per_block = (256,)
    #blocks_per_grid = (math.ceil(n/128), math.ceil(m/128), l)
    
    n_ = math.ceil(n/(128*self.patch_n))
    m_ = math.ceil(m/(128*self.patch_m))
    blocks_per_grid = (self.patch_n*self.patch_m, n_ * m_, l)
    # print(blocks_per_grid, m_, n_)

    kernel_fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        A.data_ptr(),
        B.data_ptr(),
        values.data_ptr(),
        indices.data_ptr(),
        mutex.data_ptr(),
        m, n, d, dim, 128
      ],
      stream=self.stream
    )
    indices = indices[:, :, :k]
    values = values[:, :, :k]

    if two_dimentional:
      indices = indices[0]
      values = values[0]

    return values, indices