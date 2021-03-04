import torch
import cupy as cp
import numpy as np
import math
from .CustomKernel import CustomKernel
from torchpq.util import get_absolute_path

class GetWriteAddressV2CUDA(CustomKernel):
  def __init__(
      self,
      tpb=256,
      sm_size=48*256*4,
    ):
    super(GetWriteAddressV2CUDA, self).__init__()
    self.tpb = tpb
    self.sm_size = sm_size

    with open(get_absolute_path("kernels", "GetWriteAddressV2Kernel.cu"), "r") as f:
      self.kernel = f.read()
      
    kernel = (self.kernel
      .replace("_TPB_", str(tpb))
    )

    self.fn = cp.RawKernel(
      kernel,
      'get_write_address',
      backend='nvcc',
      # options=('--maxrregcount=255',),
    )

  def __call__(self, is_empty, div_start, div_size, labels, ioa):
    """
      is_empty: [n_slots]
      div_start: [n_cluster]
      div_size: [n_cluster]
      labels: [n_labels] 
      ioa: [n_labels] index of appearance of unique label
    """
    assert div_start.shape == div_size.shape
    assert ioa.shape == labels.shape
    n_slots = is_empty.shape[0]
    n_clusters = div_start.shape[0]
    n_labels = labels.shape[0]
    
    write_adr = torch.zeros_like(labels) - 1
    threads_per_block = (self.tpb,)
    blocks_per_grid = (math.ceil(n_labels/self.tpb), )
    # print(blocks_per_grid)

    # torch.cuda.synchronize()
    self.fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        is_empty.data_ptr(),
        div_start.data_ptr(),
        div_size.data_ptr(),
        labels.data_ptr(),
        ioa.data_ptr(),
        write_adr.data_ptr(),
        n_slots, n_labels
      ],
      stream=self.stream
    )
    return write_adr