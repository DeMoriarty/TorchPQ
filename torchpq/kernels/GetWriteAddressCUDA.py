import torch
import cupy as cp
import numpy as np
import math
from .CustomKernel import CustomKernel
from torchpq.util import get_absolute_path

class GetWriteAddressCUDA(CustomKernel):
  def __init__(
      self,
      tpb=256,
      sm_size=48*256*4,
    ):
    super(GetWriteAddressCUDA, self).__init__()
    self.tpb = tpb
    self.sm_size = sm_size

    with open(get_absolute_path("kernels", "GetWriteAddressKernel.cu"), "r") as f:
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

  def __call__(self, empty_adr, div_of_empty_adr, labels, ioa):
    """
      empty_adr: [n_empty]
      div_of_empty_adr: [n_empty]
      labels: [n_labels] 
      ioa: [n_labels] index of appearance of unique label
    """
    n_empty = empty_adr.shape[0]
    n_labels = labels.shape[0]
    
    write_adr = torch.zeros_like(labels) - 1
    threads_per_block = (self.tpb,)
    blocks_per_grid = (math.ceil(n_labels/self.tpb), )

    # torch.cuda.synchronize()
    self.fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      args=[
        empty_adr.data_ptr(),
        div_of_empty_adr.data_ptr(),
        labels.data_ptr(),
        ioa.data_ptr(),
        write_adr.data_ptr(),
        n_empty, n_labels
      ],
      stream=self.stream
    )
    return write_adr