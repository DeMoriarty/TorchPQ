import torch
import cupy as cp
import numpy as np
import math
from .CustomKernel import CustomKernel
from torchpq.util import get_absolute_path

class GetDivOfAddressCUDA(CustomKernel):
  def __init__(
      self,
      ta=1,
      tpb=256,
      sm_size=48*256*4,
    ):
    super(GetDivOfAddressCUDA, self).__init__()
    self.ta = ta # how many clusters each thread is responsible of
    self.tpb = tpb
    self.sm_size = sm_size

    with open(get_absolute_path("kernels", "GetDivOfAddressKernel.cu"), "r") as f:
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

  def __call__(self, address, div_start, div_end):
    """
      address: [n_data]
      div_start: [n_clusters]
      div_end: [n_clusters]
    """
    n_data = address.shape[0]
    assert div_start.shape[0] == div_end.shape[0]
    n_clusters = div_start.shape[0]
    divs = torch.ones_like(address) * -1

    threads_per_block = (self.tpb,)
    blocks_per_grid = (math.ceil(n_clusters / (self.tpb * self.ta)), )

    # torch.cuda.synchronize()
    self.fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        address.data_ptr(),
        div_start.data_ptr(),
        div_end.data_ptr(),
        divs.data_ptr(),
        n_data, n_clusters
        ],
      stream=self.stream
    )
    return divs