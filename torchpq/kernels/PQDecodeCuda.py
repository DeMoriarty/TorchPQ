import torch
import cupy as cp
import numpy as np
import math

from .CustomKernel import CustomKernel
from ..util import get_absolute_path

class PQDecodeCuda(CustomKernel):
  def __init__(
      self,
      tm=2,
      td=8,
    ):
    super(PQDecodeCuda, self).__init__()
    self.tm = tm
    self.td = td
    self.tpb = 256
    self.sm_size = td * tm * 256 * 4

    with open(get_absolute_path("kernels", "cuda", "pq_decode.cu"), "r") as f:
      self.kernel = f.read()

    kernel = (self.kernel
      .replace("_TD_", str(td))
      .replace("_TM_", str(tm))
      .replace("_TPB_", str(self.tpb))
    )

    self.fn = cp.RawKernel(
      kernel,
      'pq_decode',
      backend="nvrtc"
    )

    self.fn.max_dynamic_shared_size_bytes = self.sm_size

  def __call__(self, codebook, code):
    """
      codebook: torch.Tensor, shape : [m, d, k], dtype : float32
      code: torch.Tensor, shape : [m, n], dtype : uint8
      return: torch.Tensor, shape : [m, d, n], dtype : float32
    """
    m, d, k = codebook.shape
    m, n = code.shape
    assert codebook.shape[0] == code.shape[0]
    assert k == 256
    device = codebook.device
    result = torch.ones(m, d, n, device=device, dtype=torch.float)

    threads_per_block = (self.tpb,)
    blocks_per_grid = (math.ceil(m/self.tm), math.ceil(d/self.td))
    self.fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      shared_mem = self.sm_size,
      args=[
        codebook.data_ptr(),
        code.data_ptr(),
        result.data_ptr(),
        m, d, n
        ],
      stream=self.stream
    )
    result = result.reshape(m*d, n)
    return result