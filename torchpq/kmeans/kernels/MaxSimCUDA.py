import torch
import cupy as cp
import numpy as np
import math
from .CustomKernel import CustomKernel
from torchpq.util import get_absolute_path

class MaxSimCUDA(CustomKernel): 
  def __init__(self, m=None, n=None, k=None, dim=None, distance="euclidean"):
    super(MaxSimCUDA, self).__init__()
    self.m = m
    self.n = n
    self.k = k
    self.dim = dim
    self.distance = distance
    with open(get_absolute_path("kmeans", "kernels","MaxSimKernel.cu"),'r') as f:
      self.kernel = f.read()

    if distance in ["euclidean", "l2"]:
      distfn = "thread_nseuclidean"
    elif distance in ["manhattan", "l1"]:
      distfn = "thread_nmanhattan"
    elif distance == "inner":
      distfn = "thread_matmul"
    elif distance == "cosine":
      print("warning: input matrices will not be normalized, please normalize them manually for cosine similarity")
      distfn = "thread_matmul"
    else:
      raise ValueError("unrecognized distance type")
      
    self.kernel = (self.kernel
      .replace("_M_", str(m) if m else "M")
      .replace("_N_", str(n) if n else "N")
      .replace("_K_", str(k) if k else "K")
      .replace("_DIM_", str(dim) if dim else "DIM")
      .replace("_DISTFN_", distfn)
    )
    
    # self._raw_module = cp.RawModule(
    #   code=self.kernel,
    #   backend='nvcc',
    #   options=('--maxrregcount=128', '--use_fast_math'),
    # )
    self._fn_tt = cp.RawKernel(
      code=self.kernel,
      name="max_sim_tt",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )
    self._fn_nn = cp.RawKernel(
      code=self.kernel,
      name="max_sim_nn",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )
    self._fn_tn = cp.RawKernel(
      code=self.kernel,
      name="max_sim_tn",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )
    self._fn_nt = cp.RawKernel(
      code=self.kernel,
      name="max_sim_nt",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )

  def _call_nn(self, A, B, dim=1):
    """
      Performs the following: 
        vals = max(A @ B, dim);
        inds = argmax(A @ B, dim)
      A: shape = [l, m, k]
      B: shape = [l, k, n]
      dim: 1 or 2. 0 is not supported.
      returns (vals, inds)
      vals: shape = [l, n] if dim=1, [l, m] if dim=2
      inds: shape = [l, n] if dim=1, [l, m] if dim=2
    """
    assert A.shape[0] == B.shape[0]
    assert A.shape[2] == B.shape[1]
    assert dim in (1, 2)
    assert A.device.type == "cuda"
    assert B.device.type == "cuda"
    assert A.dtype in (torch.float, torch.half)
    assert B.dtype in (torch.float, torch.half)
    
    l, m, k = A.shape
    l, k, n = B.shape
    
    if self.m is not None: assert m == self.m
    if self.n is not None: assert n == self.n
    if self.k is not None: assert k == self.k
    if self.dim is not None: assert dim == self.dim


    if dim == 1:
      vals = torch.full([l, n], fill_value=float("-inf"), device="cuda:0", dtype=A.dtype)
      inds = torch.empty([l, n], device="cuda:0", dtype=torch.long)
    elif dim == 2:
      vals = torch.full([l, m], fill_value=float("-inf"), device="cuda:0", dtype=A.dtype)
      inds = torch.empty([l, m], device="cuda:0", dtype=torch.long)

    threads_per_block = (256,)
    blocks_per_grid = (l, math.ceil(n/128), math.ceil(m/128))

    self._fn_nn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        A.data_ptr(),
        B.data_ptr(),
        vals.data_ptr(),
        inds.data_ptr(),
        m, n, k, dim
      ],
      stream=self.stream
    )
    return (vals, inds)

  def _call_tt(self, A, B, dim=1):
    """
      Performs the following: 
        vals = max(A.T @ B.T, dim);
        inds = argmax(A.T @ B.T, dim)
      A: shape = [l, k, m]
      B: shape = [l, n, k]
      dim: 1 or 2. 0 is not supported.
      returns (vals, inds)
      vals: shape = [l, n] if dim=1, [l, m] if dim=2
      inds: shape = [l, n] if dim=1, [l, m] if dim=2
    """
    assert A.shape[0] == B.shape[0]
    assert A.shape[1] == B.shape[2]
    assert dim in (1, 2)
    assert A.device.type == "cuda"
    assert B.device.type == "cuda"
    assert A.dtype in (torch.float, torch.half)
    assert B.dtype in (torch.float, torch.half)
    
    l, k, m = A.shape
    l, n, k = B.shape

    if self.m is not None: assert m == self.m
    if self.n is not None: assert n == self.n
    if self.k is not None: assert k == self.k
    if self.dim is not None: assert dim == self.dim


    if dim == 1:
      vals = torch.full([l, n], fill_value=float("-inf"), device="cuda:0", dtype=A.dtype)
      inds = torch.empty([l, n], device="cuda:0", dtype=torch.long)
    elif dim == 2:
      vals = torch.full([l, m], fill_value=float("-inf"), device="cuda:0", dtype=A.dtype)
      inds = torch.empty([l, m], device="cuda:0", dtype=torch.long)


    threads_per_block = (256,)
    blocks_per_grid = (l, math.ceil(n/128), math.ceil(m/128))

    self._fn_tt(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        A.data_ptr(),
        B.data_ptr(),
        vals.data_ptr(),
        inds.data_ptr(),
        m, n, k, dim
      ],
      stream=self.stream
    )
    return (vals, inds)

  def _call_tn(self, A, B, dim=1):
    """
      Performs the following: 
        vals = max(A.T @ B, dim);
        inds = argmax(A.T @ B, dim)
      A: shape = [l, k, m]
      B: shape = [l, k, n]
      dim: 1 or 2. 0 is not supported.
      returns (vals, inds)
      vals: shape = [l, n] if dim=1, [l, m] if dim=2
      inds: shape = [l, n] if dim=1, [l, m] if dim=2
    """
    assert A.shape[0] == B.shape[0]
    assert A.shape[1] == B.shape[1]
    assert dim in (1, 2)
    assert A.device.type == "cuda"
    assert B.device.type == "cuda"
    assert A.dtype in (torch.float, torch.half)
    assert B.dtype in (torch.float, torch.half)

    l, k, m = A.shape
    l, k, n = B.shape

    if self.m is not None: assert m == self.m
    if self.n is not None: assert n == self.n
    if self.k is not None: assert k == self.k
    if self.dim is not None: assert dim == self.dim

    if dim == 1:
      vals = torch.full([l, n], fill_value=float("-inf"), device="cuda:0", dtype=A.dtype)
      inds = torch.empty([l, n], device="cuda:0", dtype=torch.long)
    elif dim == 2:
      vals = torch.full([l, m], fill_value=float("-inf"), device="cuda:0", dtype=A.dtype)
      inds = torch.empty([l, m], device="cuda:0", dtype=torch.long)

    threads_per_block = (256,)
    blocks_per_grid = (l, math.ceil(n/128), math.ceil(m/128))

    self._fn_tn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        A.data_ptr(),
        B.data_ptr(),
        vals.data_ptr(),
        inds.data_ptr(),
        m, n, k, dim
      ],
      stream=self.stream,
    )
    return (vals, inds)

  def _call_nt(self, A, B, dim=1):
    """
      Performs the following: 
        vals = max(A @ B.T, dim);
        inds = argmax(A @ B.T, dim)
      A: shape = [l, m, k]
      B: shape = [l, n, k]
      dim: 1 or 2; 0 is not supported.
      returns (vals, inds)
      vals: shape = [l, n] if dim=1, [l, m] if dim=2
      inds: shape = [l, n] if dim=1, [l, m] if dim=2
    """
    assert A.shape[0] == B.shape[0]
    assert A.shape[2] == B.shape[2]
    assert dim in (1, 2)
    assert A.device.type == "cuda"
    assert B.device.type == "cuda"
    assert A.dtype in (torch.float, torch.half)
    assert B.dtype in (torch.float, torch.half)

    l, m, k = A.shape
    l, n, k = B.shape

    if self.m is not None: assert m == self.m
    if self.n is not None: assert n == self.n
    if self.k is not None: assert k == self.k
    if self.dim is not None: assert dim == self.dim

    if dim == 1:
      vals = torch.full([l, n], fill_value=float("-inf"), device="cuda:0", dtype=A.dtype)
      inds = torch.empty([l, n], device="cuda:0", dtype=torch.long)
    elif dim == 2:
      vals = torch.full([l, m], fill_value=float("-inf"), device="cuda:0", dtype=A.dtype)
      inds = torch.empty([l, m], device="cuda:0", dtype=torch.long)

    threads_per_block = (256,)
    blocks_per_grid = (l, math.ceil(n/128), math.ceil(m/128))

    self._fn_nt(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        A.data_ptr(),
        B.data_ptr(),
        vals.data_ptr(),
        inds.data_ptr(),
        m, n, k, dim
      ],
      stream=self.stream
    )
    return (vals, inds)

  def __call__(self, A, B, dim=1, mode="nn"):
    """
      Performs the following:
        vals = max(f(A) @ g(B), dim)
        inds = argmax(f(A) @ g(B), dim)
      A: torch.Tensor, shape : [l, m, k] or [l, k, m]
      B: torch.Tensor, shape : [l, n, k] or [l, k, n]
      return: (vals, inds)
      vals: torch.Tensor, shape: [l, m] if dim == 2, [l, n] if dim == 1
      inds: torch.Tensor, shape: [l, m] if dim == 2, [l, n] if dim == 1
      dim: int, default : 1
      mode: str, default: "nn"
      Notes:
        f() and g() are determined by mode
        "nn" --> A @ B
        "tt" --> A.T @ B.T
        "nt" --> A @ B.T
        "tn" --> A.T @ B
        dim should be 1 or 2, 0 is not supported.
    """
    assert len(A.shape) == len(B.shape)
    A = A.contiguous()
    B = B.contiguous()
    if len(A.shape) == 2 and len(B.shape) == 2:
      A2 = A[None]
      B2 = B[None]
      dim += 1
    elif len(A.shape) == 3 and len(B.shape) == 3:
      A2 = A
      B2 = B
    else:
      raise ValueError("shape of A and B need to be 2d or 3d")

    if mode == "nn":
      vals, inds = self._call_nn(A2, B2, dim)
    elif mode == "tt":
      vals, inds = self._call_tt(A2, B2, dim)
    elif mode == "tn":
      vals, inds = self._call_tn(A2, B2, dim)
    elif mode == "nt":
      vals, inds = self._call_nt(A2, B2, dim)

    if len(A.shape) == 2 and len(B.shape) == 2:
      vals, inds = vals[0], inds[0]
    return (vals, inds)