import torch
import cupy as cp
import math

from .CustomKernel import CustomKernel
from ..util import get_absolute_path

class IVFPQTopkCuda(CustomKernel):
  def __init__(
      self,
      m=8,
      k=256,
      tpb=256,
      n_cs=4,
      sm_size=48*256*4,
    ):
    super(IVFPQTopkCuda, self).__init__()
    assert tpb >= 32
    assert tpb == self.next_power_of_2(tpb)
    assert k == 256
    assert m * 1024 <= sm_size
    assert m % n_cs == 0
    self.m = m
    self.k = k
    self.tpb=tpb
    self.n_cs = n_cs
    self.sm_size = sm_size
    
    # with open("ivfpq_topk.cu", "r") as f:
    with open(get_absolute_path("kernels","cuda","ivfpq_topk.cu"), "r") as f:
      self.code = f.read()
    varnames = ", ".join([f"d{i}" for i in range(n_cs)])
    code = (self.code
      # .replace("_CODEBLOCK_", codeblock)
      .replace("_VARNAMES_", varnames)
      .replace("_M_", str(m))
      .replace("_K_", str(k))
      .replace("_TPB_", str(self.tpb))
      .replace("_NCS_", str(n_cs))
    )
    # print(kernel.split('\n')[60:64])
    self._topk_fn = cp.RawKernel(
      code = code,
      name = 'ivfpq_topk',
      options = (
        '--maxrregcount=128',
        '--use_fast_math'
      ),
      # backend='nvcc',
    )
    self._topk_fn.max_dynamic_shared_size_bytes = sm_size

    self._topk_residual_fn = cp.RawKernel(
      code = code,
      name = 'ivfpq_topk_residual',
      options = (
        '--maxrregcount=128',
        '--use_fast_math'
      ),
      # backend='nvcc',
    )
    self._topk_residual_fn.max_dynamic_shared_size_bytes = sm_size

    self._topk_residual_precomputed_fn = cp.RawKernel(
      code = code,
      name = 'ivfpq_topk_residual_precomputed',
      options = (
        '--maxrregcount=128',
        '--use_fast_math'
      ),
      # backend='nvcc',
    )
    self._topk_residual_precomputed_fn.max_dynamic_shared_size_bytes = sm_size
  
  @staticmethod
  def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))
  
  def topk(
      self, data, precomputed,
      is_empty, cell_start, cell_size,
      n_candidates=None
    ):
    """
      data: shape=[n_subvectors // n_cs, n_data, n_cs], dtype=uint8
      precomputed: shape=[n_query, n_clusters], dtype=float32
      is_empty: shape=[n_data], dtype=uint8
      cell_start: shape=[n_query, n_probe], dtype=int64
      cell_size: shape=[n_query, n_probe], dtype=int64
      n_candidates: int, `k` in topk
    """
    n_data = data.shape[1]
    n_query = cell_start.shape[0]
    n_probe = cell_start.shape[1]
    assert precomputed.shape == (self.m, n_query, self.k)
    assert data.shape[0] == self.m // self.n_cs
    assert data.shape[2] == self.n_cs
    assert is_empty.shape[0] == n_data
    assert cell_size.shape[1] == n_probe
    assert data.dtype == torch.uint8
    assert precomputed.dtype == torch.float32
    assert cell_start.dtype == cell_size.dtype == torch.int64
    assert is_empty.dtype == torch.uint8
    if n_candidates is None:
      n_candidates = self.tpb
    else:
      assert n_candidates <= self.tpb
    n_candidates_pow_of_2 = 2 * self.next_power_of_2(math.ceil(n_candidates / 2))
    assert n_candidates_pow_of_2 in [2 * 2**i for i in range(10)]

    tot_size = cell_size.sum(dim=1)
    values = torch.empty(n_query, n_candidates_pow_of_2, device="cuda:0", dtype=torch.float32)
    values.fill_(float("-inf"))
    indices = torch.zeros(n_query, n_candidates_pow_of_2, device="cuda:0", dtype=torch.int64)
    threads_per_block = (self.tpb,)
    blocks_per_grid = (n_query,)

    self._topk_fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      shared_mem = self.sm_size,
      args=[
        data.data_ptr(),
        precomputed.data_ptr(),
        is_empty.data_ptr(),
        cell_start.data_ptr(),
        cell_size.data_ptr(),
        tot_size.data_ptr(),
        values.data_ptr(),
        indices.data_ptr(),
        n_data, n_query, n_probe, n_candidates_pow_of_2
        ],
      stream=self.stream
    )
    return (values[:, :n_candidates], indices[:, :n_candidates])

  def topk_residual(
      self, data, precomputed,
      is_empty, cell_start, cell_size,
      base_sims, n_candidates=None
    ):
    """
      data: shape=[n_subvectors // n_cs, n_data, n_cs], dtype=uint8
      precomputed: shape=[n_query, n_probe, n_subvectors, n_clusters], dtype=float32
      is_empty: shape=[n_data], dtype=uint8
      cell_start: shape=[n_query, n_probe], dtype=int64
      cell_size: shape=[n_query, n_probe], dtype=int64
      base_sims: shape=[n_query, n_probe], dtype=float32
      n_candidates: int, `k` in topk
    """
    n_data = data.shape[1]
    n_query = cell_start.shape[0]
    n_probe = cell_start.shape[1]
    assert precomputed.shape == (n_query, n_probe, self.m, self.k)
    assert data.shape == (self.m // self.n_cs, n_data, self.n_cs)
    assert is_empty.shape == (n_data,)
    assert cell_size.shape == (n_query, n_probe)
    assert base_sims.shape == (n_query, n_probe)
    assert data.dtype == torch.uint8
    assert precomputed.dtype == torch.float32
    assert cell_start.dtype == cell_size.dtype == torch.int64
    assert is_empty.dtype == torch.uint8
    assert base_sims.dtype == torch.float32
    precomputed = precomputed.contiguous()
    base_sims = base_sims.contiguous()
    if n_candidates is None:
      n_candidates = self.tpb
    else:
      assert n_candidates <= self.tpb
    n_candidates_pow_of_2 = 2 * self.next_power_of_2(math.ceil(n_candidates / 2))
    assert n_candidates_pow_of_2 in [2 * 2**i for i in range(10)]

    tot_size = cell_size.sum(dim=1)
    values = torch.empty(n_query, n_candidates_pow_of_2, device="cuda:0", dtype=torch.float32)
    values.fill_(float("-inf"))
    indices = torch.zeros(n_query, n_candidates_pow_of_2, device="cuda:0", dtype=torch.int64)
    threads_per_block = (self.tpb,)
    blocks_per_grid = (n_query,)

    self._topk_residual_fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      shared_mem = self.sm_size,
      args=[
        data.data_ptr(),
        precomputed.data_ptr(),
        base_sims.data_ptr(),
        is_empty.data_ptr(),
        cell_start.data_ptr(),
        cell_size.data_ptr(),
        tot_size.data_ptr(),
        values.data_ptr(),
        indices.data_ptr(),
        n_data, n_query, n_probe, n_candidates_pow_of_2
        ],
      stream=self.stream
    )
    return (values[:, :n_candidates], indices[:, :n_candidates])

  def topk_residual_precomputed(
      self, data, part1, part2, cells, base_sims,
      is_empty, cell_start, cell_size,
      n_candidates=None
    ):
    """
      data: shape=[n_subvectors // n_cs, n_data, n_cs], dtype=uint8
      part1: shape=[n_query, n_subvectors, n_pq_clusters], dtype=float32
      part2: shape=[n_cells, n_subvectors, n_pq_clusters], dtype=float32
      cells: shape=[n_query, n_probe], dtype=int64
      base_sims: shape=[n_query, n_probe], dtype=float32
      is_empty: shape=[n_data], dtype=uint8
      cell_start: shape=[n_query, n_probe], dtype=int64
      cell_size: shape=[n_query, n_probe], dtype=int64
      n_candidates: int, `k` in topk
    """
    n_data = data.shape[1]
    n_query = cell_start.shape[0]
    n_probe = cell_start.shape[1]
    assert data.shape == (self.m // self.n_cs, n_data, self.n_cs)
    assert is_empty.shape == (n_data,)
    assert cell_size.shape == (n_query, n_probe)
    assert base_sims.shape == (n_query, n_probe)
    assert data.dtype == torch.uint8
    assert part1.dtype == part2.dtype == torch.float32
    assert cell_start.dtype == cell_size.dtype == torch.int64
    assert is_empty.dtype == torch.uint8
    assert base_sims.dtype == torch.float32
    part1 = part1.contiguous()
    part2 = part2.contiguous()
    cells = cells.contiguous()
    base_sims = base_sims.contiguous()
    if n_candidates is None:
      n_candidates = self.tpb
    else:
      assert n_candidates <= self.tpb
    n_candidates_pow_of_2 = 2 * self.next_power_of_2(math.ceil(n_candidates / 2))
    assert n_candidates_pow_of_2 in [2 * 2**i for i in range(10)]

    tot_size = cell_size.sum(dim=1)
    values = torch.empty(n_query, n_candidates_pow_of_2, device="cuda:0", dtype=torch.float32)
    values.fill_(float("-inf"))
    indices = torch.zeros(n_query, n_candidates_pow_of_2, device="cuda:0", dtype=torch.int64)
    threads_per_block = (self.tpb,)
    blocks_per_grid = (n_query,)

    self._topk_residual_precomputed_fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      shared_mem = self.sm_size,
      args=[
        data.data_ptr(),
        part1.data_ptr(),
        part2.data_ptr(),
        cells.data_ptr(),
        base_sims.data_ptr(),
        is_empty.data_ptr(),
        cell_start.data_ptr(),
        cell_size.data_ptr(),
        tot_size.data_ptr(),
        values.data_ptr(),
        indices.data_ptr(),
        n_data, n_query, n_probe, n_candidates_pow_of_2
        ],
      stream=self.stream
    )
    return (values[:, :n_candidates], indices[:, :n_candidates])
