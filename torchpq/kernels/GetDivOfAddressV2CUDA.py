import torch
import cupy as cp
import numpy as np
import math
from .CustomKernel import CustomKernel
from torchpq.util import get_absolute_path

class GetDivOfAddressV2CUDA(CustomKernel):
  def __init__(
      self,
      ta=1,
      tpb=256,
      sm_size=48*256*4,
    ):
    super(GetDivOfAddressV2CUDA, self).__init__()
    self.ta = ta # how many clusters each thread is responsible of
    self.tpb = tpb
    self.sm_size = sm_size
    assert ta * tpb * 8 <= sm_size

    with open(get_absolute_path("kernels", "GetDivOfAddressV2Kernel.cu"), "r") as f:
      self.kernel = f.read()
    kernel = (self.kernel
      .replace("_TA_", str(ta))
      .replace("_TPB_", str(tpb))
    )

    self.fn = cp.RawKernel(
      kernel,
      'get_div_of_address',
      backend='nvcc',
      # options=('--maxrregcount=255',),
    )
    self.fn.max_dynamic_shared_size_bytes = ta * tpb * 8

  def __call__(self, address, div_start, div_end):
    """
      address: [n_address]
      div_start: [n_divs]
      div_end: [n_divs]
    """
    n_address = address.shape[0]
    assert div_start.shape[0] == div_end.shape[0]
    n_divs = div_start.shape[0]
    sorted_address, sorted_address_index = address.sort()
    out = torch.ones_like(address) * -1
    divs = torch.zeros_like(address)

    threads_per_block = (self.tpb,)
    blocks_per_grid = (math.ceil(n_address / (self.tpb * self.ta)), )

    # torch.cuda.synchronize()
    self.fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      shared_mem = self.ta * self.tpb * 8,
      args=[
        sorted_address.data_ptr(),
        div_start.data_ptr(),
        div_end.data_ptr(),
        out.data_ptr(),
        n_address, n_divs
        ],
      stream=self.stream
    )
    divs[sorted_address_index] = out
    return divs