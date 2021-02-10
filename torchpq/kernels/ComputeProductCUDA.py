import torch
import cupy as cp
import numpy as np
import math
from .CustomKernel import CustomKernel
from torchpq.util import get_absolute_path

class ComputeProductCUDA(CustomKernel):
  def __init__(
      self,
      m=8,
      k=256,
      n_cs=4,
      sm_size=48*256*4,
    ):
    super(ComputeProductCUDA, self).__init__()
    self.m = m
    self.k = k
    self.tpb = 256
    self.n_cs = n_cs
    self.sm_size = sm_size

    with open(get_absolute_path("kernels", "ComputeProductKernel.cu"), "r") as f:
      self.kernel = f.read()
    
    cb1 = [f"      float Bval{i} = Bsh[(i * _NCS_ + {i}) * _K_ + int(Avals.d{i}) ];" for i in range(n_cs)]
    cb2 = [f"      sum += Bval{i};" for i in range(n_cs)]
    codeblock = "\n".join(cb1) + "\n" + "\n".join(cb2)
    varnames = ", ".join([f"d{i}" for i in range(n_cs)])
    kernel = (self.kernel
      .replace("_CODEBLOCK_", codeblock)
      .replace("_VARNAMES_", varnames)
      .replace("_M_", str(m))
      .replace("_K_", str(k))
      .replace("_TPB_", str(self.tpb))
      .replace("_NCS_", str(n_cs))
    )
    # print(kernel.split('\n')[60:64])
    self.fn = cp.RawKernel(
      kernel,
      'compute_product',
      # options=('--maxrregcount=255',),
      # backend='nvcc',
    )

    self.fn.max_dynamic_shared_size_bytes = sm_size
    # print(self.fn.attributes)

  def __call__(self, data, precomputed, is_empty, div_start, div_size, max_out_size=None):
    """
      data: shape=[n_subvectors // n_cs, n_data, n_cs], dtype=uint8
      precomputed: shape=[n_subvectors, n_query, n_clusters], dtype=float32
      is_empty: shape=[n_data], dtype=uint8
      div_start: shape=[n_query, n_probe], dtype=int32
      div_size: shape=[n_query, n_probe], dtype=int32
    """
    n_data = data.shape[1]
    n_query = precomputed.shape[1]
    n_probe = div_start.shape[1]
    assert precomputed.shape[0] == self.m
    assert precomputed.shape[2] == self.k
    assert data.shape[0] == self.m // self.n_cs
    assert data.shape[2] == self.n_cs
    assert is_empty.shape[0] == n_data
    assert div_size.shape[1] == n_probe
    assert data.dtype == torch.uint8
    assert precomputed.dtype == torch.float32
    assert div_start.dtype == div_size.dtype == torch.int32
    assert is_empty.dtype == torch.uint8

    if max_out_size is None:
      max_out_size = div_size.sum(dim=1).max().item()
    values = torch.empty(n_query, max_out_size, device="cuda:0", dtype=torch.float32)
    values.fill_(float("-inf"))
    indices = torch.zeros(n_query, max_out_size, device="cuda:0", dtype=torch.int32) #?

    threads_per_block = (self.tpb,)
    blocks_per_grid = (n_query,)
    self.fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      shared_mem = self.sm_size,
      args=[
        data.data_ptr(),
        precomputed.data_ptr(),
        is_empty.data_ptr(),
        div_start.data_ptr(),
        div_size.data_ptr(),
        values.data_ptr(),
        indices.data_ptr(),
        n_data, n_query, max_out_size, n_probe,
        ],
      stream=self.stream
    )
    return (values, indices)