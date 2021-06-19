import torch
import cupy as cp

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
    
    with open(get_absolute_path("kernels", "cuda", "ivfpq_topk.cu"), "r") as f:
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
    self.fn = cp.RawKernel(
      code = kernel,
      name = 'ivfpq_topk',
      options = (
        '--maxrregcount=128',
        '--use_fast_math'
      ),
      # backend='nvcc',
    )

    self.fn.max_dynamic_shared_size_bytes = sm_size
  
  @staticmethod
  def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))
  
  def __call__(
      self, data, precomputed,
      is_empty, div_start, div_size,
      n_candidates=None
    ):
    """
      data: shape=[n_subvectors // n_cs, n_data, n_cs], dtype=uint8
      precomputed: shape=[n_subvectors, n_query, n_clusters], dtype=float32
      is_empty: shape=[n_data], dtype=uint8
      div_start: shape=[n_query, n_probe], dtype=int32
      div_size: shape=[n_query, n_probe], dtype=int32
      n_candidates: int, `k` in topk
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
    assert div_start.dtype == div_size.dtype == torch.int64
    assert is_empty.dtype == torch.uint8
    if n_candidates is None:
      n_candidates = self.tpb
    else:
      assert n_candidates <= self.tpb
    n_candidates_pow_of_2 = 2 * self.next_power_of_2(math.ceil(n_candidates / 2))
    assert n_candidates_pow_of_2 in [2 * 2**i for i in range(10)]

    tot_size = div_size.sum(dim=1)
    values = torch.empty(n_query, n_candidates_pow_of_2, device="cuda:0", dtype=torch.float32)
    values.fill_(float("-inf"))
    indices = torch.zeros(n_query, n_candidates_pow_of_2, device="cuda:0", dtype=torch.int64)
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
        tot_size.data_ptr(),
        values.data_ptr(),
        indices.data_ptr(),
        n_data, n_query, n_probe, n_candidates_pow_of_2
        ],
      stream=self.stream
    )
    return (values[:, :n_candidates], indices[:, :n_candidates])
