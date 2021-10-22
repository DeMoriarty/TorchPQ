import torch
import cupy as cp
import numpy as np
import math

from .CustomKernel import CustomKernel
from ..util import get_absolute_path

class GetAddressByIDCuda(CustomKernel):
  def __init__(
      self,
      tpb=256,
      sm_size=48*256*4,
    ):
    super(GetAddressByIDCuda, self).__init__()
    self.tpb = tpb
    self.sm_size = sm_size

    with open(get_absolute_path("kernels", "cuda","get_address_by_id.cu"), "r") as f:
      self.kernel = f.read()
    kernel = (self.kernel
      .replace("_TPB_", str(tpb))
    )

    self.fn = cp.RawKernel(
      kernel,
      'get_address_by_id',
      backend='nvrtc',
      # options=('--maxrregcount=255',),
    )

  def __call__(self, address2id, ids):
    """
      address2id: [n_data]
      ids: [n_ids]
    """
    n_data = address2id.shape[0]
    n_ids = ids.shape[0]
    address = torch.ones_like(ids) * -1

    threads_per_block = (self.tpb,)
    blocks_per_grid = (math.ceil(n_ids/self.tpb), )

    # torch.cuda.synchronize()
    self.fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        address2id.data_ptr(),
        ids.data_ptr(),
        address.data_ptr(),
        n_data, n_ids
        ],
      stream=self.stream
    )
    return address