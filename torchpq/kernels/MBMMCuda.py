import torch
import cupy as cp
import numpy as np
import math
from custom_kernel import CustomKernel

class MBMMCuda(CustomKernel): 
  def __init__(self,
      patch_m = 4,
      patch_n = 4,
      share_mask=False,
    ):
    super(MBMMCuda, self).__init__()
    assert type(share_mask) == bool
    self.patch_m = patch_m
    self.patch_n = patch_n
    self.share_mask = share_mask
    with open("kernels/mbmm_kernel.cu",'r') as f: ###
      self.kernel = f.read()
      
    self.kernel = (self.kernel
      .replace("_PM_", str(self.patch_m))
      .replace("_PN_", str(self.patch_n))
      .replace("__MASK_BID__", "0" if share_mask else "bid")
    )
    
    self._fn_tt = cp.RawKernel(
      code=self.kernel,
      name="mbmm_tt",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )
    self._fn_nn = cp.RawKernel(
      code=self.kernel,
      name="mbmm_nn",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )
    self._fn_tn = cp.RawKernel(
      code=self.kernel,
      name="mbmm_tn",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )
    self._fn_nt = cp.RawKernel(
      code=self.kernel,
      name="mbmm_nt",
      backend='nvcc',
      options=('--maxrregcount=128', '--use_fast_math')
    )

  def _call_nn(self, A, B, block_mask, thread_mask, element_mask):
    assert A.shape[0] == B.shape[0]
    assert A.shape[2] == B.shape[1]
    assert A.device.type == "cuda"
    assert B.device.type == "cuda"
    assert A.dtype in (torch.float, torch.half)
    assert B.dtype in (torch.float, torch.half)
    
    l, m, k = A.shape
    l, k, n = B.shape
    assert block_mask.dtype == torch.uint8 ###
    assert thread_mask.dtype == torch.uint8 ###
    assert element_mask.dtype == torch.uint8
    if self.share_mask:
      assert block_mask.shape == (math.ceil(m / 128), math.ceil(n / 128))  ###
      assert thread_mask.shape == (math.ceil(m / 8), math.ceil(n / 8)) ###
      assert element_mask.shape == (m, n)
    else:
      assert block_mask.shape == (l, math.ceil(m / 128), math.ceil(n / 128))  ###
      assert thread_mask.shape == (l, math.ceil(m / 8), math.ceil(n / 8)) ###
      assert element_mask.shape == (l, m, n)
    
    C = torch.zeros(l, m, n, device="cuda:0", dtype=A.dtype)

    threads_per_block = (256,)

    # blocks_per_grid = (l, math.ceil(n/128), math.ceil(m/128))
    n_ = math.ceil(n / (128 * self.patch_n))
    m_ = math.ceil(m / (128 * self.patch_m))
    blocks_per_grid = (self.patch_n * self.patch_m, n_ * m_, l)


    self._fn_nn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        block_mask.data_ptr(),
        thread_mask.data_ptr(),
        element_mask.data_ptr(),
        m, n, k
      ],
      stream=self.stream
    )
    return C

  def _call_tt(self, A, B, block_mask, thread_mask, element_mask):
    raise NotImplementedError

  def _call_tn(self, A, B, block_mask, thread_mask, element_mask):
    raise NotImplementedError

  def _call_nt(self, A, B, block_mask, thread_mask, element_mask):
    raise NotImplementedError

  def __call__(
      self,
      A,
      B,
      block_mask,
      thread_mask,
      element_mask,
      mode="nn"
    ):
    """
      Performs C = f(A) @ g(B)
      A:
        torch.Tensor
        shape : [m, k] or [k, m] or [l, m, k] or [l, k, m]
        dtype : float32

      B:
        torch.Tensor
        shape : [n, k] or [k, n] or [l, n, k] or [l, k, n]
        dtype : float32

      element_mask:
        mask of elements in C that are not computed
        torch.Tensor, dtype : uint8
        if *share_mask* == True
          shape : [m, n]
        else
          shape : [l, m, n]

      block_mask:
        mask of 128x128 blocks in C that are not computed
        torch.Tensor
        dtype : uint8
        if *share_mask* == True
          shape : [ceil(m/128), ceil(n/128)]
        else
          shape : [l, ceil(m/128), ceil(n/128)]

      thread_mask:
        mask of 8x8 blocks in C that are not computed
        torch.Tensor
        dtype : uint8
        if *share_mask* == True
          shape : [ceil(m/8), ceil(n/8)]
        else
          shape : [l, ceil(m/8), ceil(n/8)]

      mode: {"nn", "tn", "nt", "tt"}, default: "nn"

      returns C:
        torch.Tensor
        shape : [m, n] or [l, m, n]
        dtype : float32

      Notes:
        f() and g() are determined by *mode*
        "nn" --> A @ B
        "tt" --> A.T @ B.T
        "nt" --> A @ B.T
        "tn" --> A.T @ B
    """
    assert len(A.shape) == len(B.shape)
    A = A.contiguous()
    B = B.contiguous()
    if len(A.shape) == 2 and len(B.shape) == 2:
      A2 = A[None]
      B2 = B[None]
      if not self.share_mask:
        block_mask = block_mask[None]
        thread_mask = thread_mask[None]
        element_mask = element_mask[None]
    elif len(A.shape) == 3 and len(B.shape) == 3:
      A2 = A
      B2 = B
    else:
      raise ValueError("shape of A and B need to be 2d or 3d")

    if mode == "nn":
      C = self._call_nn(A2, B2, block_mask, thread_mask, element_mask)
    elif mode == "tt":
      C = self._call_tt(A2, B2, block_mask, thread_mask, element_mask)
    elif mode == "tn":
      C = self._call_tn(A2, B2, block_mask, thread_mask, element_mask)
    elif mode == "nt":
      C = self._call_nt(A2, B2, block_mask, thread_mask, element_mask)

    if len(A.shape) == 2 and len(B.shape) == 2:
      C = C[0]
    return C