import torch
import cupy as cp
import numpy as np
import math
from .CustomKernel import CustomKernel
from torchpq.util import get_absolute_path

class GetIOACUDA(CustomKernel):
  def __init__(
      self,
      tpb=256,
      sm_size=48*256*4,
    ):
    super(GetIOACUDA, self).__init__()
    self.tpb = tpb
    self.sm_size = sm_size

    with open(get_absolute_path("kernels", "GetIOAKernel.cu"), "r") as f:
      self.kernel = f.read()
    kernel = (self.kernel
      .replace("_TPB_", str(tpb))
    )

    self.fn = cp.RawKernel(
      kernel,
      'get_ioa',
      backend='nvcc',
      # options=('--maxrregcount=255',),
    )

  def __call__(self, labels, unique_labels=None):
    """
      labels: [n_labels]
      unique_labels: [n_unique]
    """
    n_labels = labels.shape[0]
    if unique_labels is None:
      unique_labels = torch.unique(labels)
    n_unique = unique_labels.shape[0]

    ioa = torch.zeros_like(labels) - 2
    threads_per_block = (self.tpb,)
    blocks_per_grid = (math.ceil(n_unique / self.tpb), )

    # torch.cuda.synchronize()
    self.fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        labels.data_ptr(),
        unique_labels.data_ptr(),
        ioa.data_ptr(),
        n_labels, n_unique
        ],
      stream=self.stream
    )
    return ioa