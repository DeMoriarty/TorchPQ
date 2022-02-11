import torch
import cupy as cp
import math

from .CustomKernel import CustomKernel
from ..util import get_absolute_path
from time import time

class DistributedIVFPQTop1Cuda(CustomKernel):
  """
    What's new:
      bunch of stuff
  """
  def __init__(
      self,
      m=8,
      k=256,
      tpb=256,
      n_cs=4,
      sm_size=48*256*4,
    ):
    super().__init__()
    assert tpb >= 32
    assert tpb == self.next_power_of_2(tpb), "tpb needs to be a power of 2"
    assert k == 256
    assert m * 1024 <= sm_size
    assert m % n_cs == 0
    self.m = m
    self.k = k
    self.tpb=tpb
    self.n_cs = n_cs
    self.sm_size = sm_size
    
    with open(get_absolute_path("kernels", "cuda", "distributed_ivfpq_top1.cu"), "r") as f:
      self.kernel = f.read()
    varnames = ", ".join([f"d{i}" for i in range(n_cs)])
    kernel = (self.kernel
      # .replace("_CODEBLOCK_", codeblock)
      .replace("_VARNAMES_", varnames)
      .replace("_M_", str(m))
      .replace("_K_", str(k))
      .replace("_TPB_", str(self.tpb))
      .replace("_NCS_", str(n_cs))
    )
    # print(kernel.split('\n')[60:64])
    self._top1_fn = cp.RawKernel(
      code = kernel,
      name = f'ivfpq_top1',
      options = (
        '--maxrregcount=255',
        '--use_fast_math',
        '-lineinfo'
      ),
      backend='nvrtc',
    )
    self._top1_fn.max_dynamic_shared_size_bytes = sm_size

    self._top1_residual_fn = cp.RawKernel(
      code = kernel,
      name = f'ivfpq_top1_residual',
      options = (
        '--maxrregcount=255',
        '--use_fast_math',
        '-lineinfo'
      ),
      backend='nvrtc',
    )
    self._top1_residual_fn.max_dynamic_shared_size_bytes = sm_size
    
    self._top1_residual_precomputed_fn = cp.RawKernel(
      code = kernel,
      name = 'ivfpq_top1_residual_precomputed',
      options = (
        '--maxrregcount=255',
        '--use_fast_math',
        '-lineinfo'
      ),
      backend='nvrtc',
    )
    self._top1_residual_precomputed_fn.max_dynamic_shared_size_bytes = sm_size

  @staticmethod
  def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))
  
  def topk(
      self,
      address2id_ptr,
      precomputed,
      cell_ptr, 
      cell_size, 
      cell_capacity,
      n_probe_list, 
      n_candidates=1
    ):
    """
      # data: shape=[n_subvectors // n_cs, n_data, n_cs], dtype=uint8
      # is_empty: shape=[n_data], dtype=uint8
      address2id_ptr: shape=[n_query, max_n_probe]
      precomputed: shape=[n_subvectors, n_query, n_clusters], dtype=float32
      cell_ptr: shape=[n_query, max_n_probe], dtype=int64
      cell_size: shape=[n_query, max_n_probe], dtype=int64
      cell_capacity: shape=[n_query, max_n_probe], dtype=int64
      n_probe_list: shape=[n_query], dtype=int64
    """
    n_query = precomputed.shape[1]
    n_probe = cell_ptr.shape[1]
    assert precomputed.shape[0] == self.m
    assert precomputed.shape[2] == self.k
    assert precomputed.dtype == torch.float32
    # n_data = data.shape[1]
    # assert data.shape[0] == self.m // self.n_cs
    # assert data.shape[2] == self.n_cs
    # assert data.dtype == torch.uint8
    # assert is_empty.shape[0] == n_data
    # assert is_empty.dtype == torch.uint8
    assert cell_size.shape[1] == n_probe
    assert cell_ptr.dtype == cell_size.dtype == cell_capacity.dtype == torch.int64
    assert n_probe_list.shape == (n_query, )
    assert n_probe_list.dtype == torch.int64
    assert n_candidates == 1
    device = precomputed.device

    cell_info = torch.stack([cell_size, cell_ptr, cell_capacity], dim=-1)
    tot_size = cell_size.sum(dim=1)
    values = torch.empty(n_query, 1, device=device, dtype=torch.float32)
    values.fill_(float("-inf"))
    address = torch.zeros(n_query, 1, 2, device=device, dtype=torch.int64)
    ids = torch.zeros(n_query, 1, device=device, dtype=torch.int64)
    threads_per_block = (self.tpb,)
    blocks_per_grid = (n_query,)

    self._top1_fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      shared_mem = self.sm_size,
      args=[
        # data.data_ptr(),
        address2id_ptr.data_ptr(),
        precomputed.data_ptr(),
        cell_info.data_ptr(),
        tot_size.data_ptr(),
        n_probe_list.data_ptr(),
        values.data_ptr(),
        address.data_ptr(),
        ids.data_ptr(),
        n_query, n_probe
        ],
      stream=self.stream
    )
    return (values, address)

  def topk_residual(
      self,
      address2id_ptr,
      precomputed, 
      base_sims,
      cell_ptr, 
      cell_size, 
      cell_capacity, 
      n_probe_list, 
      n_candidates=1
    ):
    """
      address2id_ptr: shape=[n_query, max_n_probe], dtype=int64
      precomputed: shape=[n_query, max_n_probe, n_subvectors, n_clusters], dtype=float32
      base_sims: shape=[n_query, max_n_probe], dtype=float32
      cell_ptr: shape=[n_query, max_n_probe], dtype=int64
      cell_size: shape=[n_query, max_n_probe], dtype=int64
      cell_capacity: shape=[n_query, max_n_probe], dtype=int64
      n_probe_list: shape=[n_query], dtype=int64
    """
    # n_data = data.shape[1]
    n_query = cell_ptr.shape[0]
    n_probe = cell_ptr.shape[1]
    assert precomputed.shape == (n_query, n_probe, self.m, self.k)
    assert base_sims.shape == (n_query, n_probe)
    # assert data.shape == (self.m // self.n_cs, n_data, self.n_cs)
    # assert data.dtype == torch.uint8
    # assert is_empty.shape == (n_data, )
    # assert is_empty.dtype == torch.uint8
    assert cell_size.shape == (n_query, n_probe)
    assert cell_ptr.shape == (n_query, n_probe)
    assert cell_capacity.shape == (n_query, n_probe)
    assert precomputed.dtype == torch.float32
    assert cell_ptr.dtype == cell_size.dtype == cell_capacity.dtype == torch.int64
    assert base_sims.dtype == torch.float32
    assert n_candidates == 1
    assert n_probe_list.shape == (n_query, )
    assert n_probe_list.dtype == torch.int64
    base_sims = base_sims.contiguous()
    device = address2id_ptr.device

    cell_info = torch.stack([cell_size, cell_ptr, cell_capacity], dim=-1)
    tot_size = cell_size.sum(dim=1)
    values = torch.empty(n_query, 1, device=device, dtype=torch.float32)
    values.fill_(float("-inf"))
    indices = torch.zeros(n_query, 1, device=device, dtype=torch.int64)
    threads_per_block = (self.tpb,)
    blocks_per_grid = (n_query,)

    self._top1_residual_fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      shared_mem = self.sm_size,
      args=[
        address2id_ptr.data_ptr(),
        precomputed.data_ptr(),
        base_sims.data_ptr(),
        cell_info.data_ptr(),
        tot_size.data_ptr(),
        n_probe_list.data_ptr(),
        values.data_ptr(),
        indices.data_ptr(),
        n_query, n_probe
        ],
      stream=self.stream
    )
    return (values, indices)

  def topk_residual_precomputed(
      self,
      address2id_ptr,
      part1, 
      part2, 
      cells, 
      base_sims,
      cell_ptr, 
      cell_size, 
      cell_capacity, 
      n_probe_list, 
      n_candidates=1
    ):
    """
      address2id_ptr: shape=[n_query, max_n_probe], dtype=int64
      part1: shape=[n_query, n_subvectors, n_pq_clusters], dtype=float32
      part2: shape=[n_cells, n_subvectors, n_pq_clusters], dtype=float32
      cells: shape=[n_query, max_n_probe], dtype=int64
      base_sims: shape=[n_query, max_n_probe], dtype=float32
      cell_ptr: shape=[n_query, max_n_probe], dtype=int64
      cell_size: shape=[n_query, max_n_probe], dtype=int64
      cell_capacity: shape=[n_query, max_n_probe], dtype=int64
      n_probe_list: shape=[n_query], dtype=int64
    """
    # n_data = data.shape[1]
    n_query = cell_ptr.shape[0]
    n_probe = cell_ptr.shape[1]
    assert base_sims.shape == (n_query, n_probe)
    assert cells.shape == (n_query, n_probe)
    assert address2id_ptr.shape == (n_query, n_probe)
    # assert data.shape == (self.m // self.n_cs, n_data, self.n_cs)
    # assert data.dtype == torch.uint8
    # assert is_empty.shape == (n_data, )
    # assert is_empty.dtype == torch.uint8
    assert cell_size.shape == (n_query, n_probe)
    assert cell_ptr.shape == (n_query, n_probe)
    assert cell_capacity.shape == (n_query, n_probe)
    assert cell_ptr.dtype == cell_size.dtype == cell_capacity.dtype == torch.int64
    assert base_sims.dtype == torch.float32
    assert cells.dtype == torch.int64
    assert part1.dtype == part2.dtype == torch.float32
    assert n_candidates == 1
    assert n_probe_list.shape == (n_query, )
    assert n_probe_list.dtype == torch.int64
    part1 = part1.contiguous()
    part2 = part2.contiguous()
    cells = cells.contiguous()
    base_sims = base_sims.contiguous()
    device = address2id_ptr.device

    cell_info = torch.stack([cell_size, cell_ptr, cell_capacity], dim=-1)
    tot_size = cell_size.sum(dim=1)
    values = torch.empty(n_query, 1, device=device, dtype=torch.float32)
    values.fill_(float("-inf"))
    indices = torch.zeros(n_query, 1, device=device, dtype=torch.int64)
    threads_per_block = (self.tpb,)
    blocks_per_grid = (n_query,)

    self._top1_residual_precomputed_fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      shared_mem = self.sm_size,
      args=[
        address2id_ptr.data_ptr(),
        part1.data_ptr(),
        part2.data_ptr(),
        cells.data_ptr(),
        base_sims.data_ptr(),
        cell_info.data_ptr(),
        tot_size.data_ptr(),
        n_probe_list.data_ptr(),
        values.data_ptr(),
        indices.data_ptr(),
        n_query, n_probe
        ],
      stream=self.stream
    )
    return (values, indices)
