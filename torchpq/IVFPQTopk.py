import torch
import numpy as np
import math
from .kernels import ComputeProductCUDA

class IVFPQTopk:
  def __init__(self,
    n_subvectors,
    n_clusters,
    n_cs=4,
    ):
    assert torch.cuda.is_available()
    self.n_subvectors = n_subvectors
    self.n_clusters = n_clusters
    self.n_cs = n_cs
    self.sm_size = n_subvectors * 256 * 4

    self.compute_product = ComputeProductCUDA(
      m=n_subvectors,
      k=n_clusters,
      n_cs=n_cs,
      sm_size=self.sm_size
    )

  @staticmethod
  def remaining_memory():
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
      total_memory = torch.cuda.get_device_properties(0).total_memory
      remaining = total_memory - torch.cuda.memory_reserved()
    else:
      remaining = 0
    return remaining

  def get_similarity(self, data, precomputed, is_empty, div_start, div_size):
    max_out_size = div_size.sum(dim=1).max().item()
    n_subvectors, n_query, n_clusters = precomputed.shape
    n_probe = div_start.shape[1]

    values, indices = self.compute_product(
      data = data,
      precomputed = precomputed,
      is_empty = is_empty,
      div_start = div_start,
      div_size = div_size,
      max_out_size = max_out_size,  
    )
    return values, indices

  def __call__(self, k, data, precomputed, is_empty, div_start, div_size):
    """
      k: dtype : int
      data: shape : [n_subvectors // n_cs, n_data, n_cs], dtype : uint8
      precomputed: shape : [n_subvectors, n_query, n_clusters], dtype : float32
      is_empty: shape : [n_data], dtype : uint8
      div_start: shape : [n_query, n_probe], dtype : int32
      div_size: shape : [n_query, n_probe], dtype : int32
    """
    max_out_size = div_size.sum(dim=1).max().item()
    n_subvectors, n_query, n_clusters = precomputed.shape
    n_probe = div_start.shape[1]


    final_v = torch.zeros(n_query, k, device="cuda:0", dtype=torch.float32)
    final_i = torch.zeros(n_query, k, device="cuda:0", dtype=torch.int32)
    remaining = self.remaining_memory()
    n_partitions = 1
    while True:
      if n_partitions > n_query:
        raise RuntimeError("No enough GPU memory")
      sub_n_query = math.ceil(n_query / n_partitions)
      required = sub_n_query * max_out_size * 2 * 4
      if n_partitions > 1:
        required += sub_n_query * n_subvectors * n_clusters * 4
        required += sub_n_query * n_probe * 2 * 4
      if required <= remaining:
        break
      n_partitions *= 2
    for i in range(n_partitions):
      start = i * sub_n_query
      end = (i+1) * sub_n_query
      if end > n_query:
        end = n_query
      if n_partitions > 1:
        sub_precomputed = precomputed[:, start:end].contiguous()
        sub_div_start = div_start[start:end].contiguous()
        sub_div_size = div_size[start:end].contiguous()
        sub_mos = sub_div_size.sum(dim=1).max().item()
      else:
        sub_precomputed = precomputed
        sub_div_start = div_start
        sub_div_size = div_size
        sub_mos = max_out_size

      sub_v, sub_i = self.compute_product(
        data = data,
        precomputed = sub_precomputed,
        is_empty = is_empty,
        div_start = sub_div_start,
        div_size = sub_div_size,
        max_out_size = sub_mos,  
      )
      del sub_precomputed
      sub_k = min(k, sub_mos)
      sorted_v, sorted_i = torch.topk(sub_v, dim=-1, k=sub_k)
      del sub_v
      final_v[start:end, :sub_k] = sorted_v
      del sorted_v
      final_i[start:end, :sub_k] = torch.gather(input=sub_i, index=sorted_i, dim=1)
      del sub_i, sorted_i

    ### TEST
    # def naive_pqd(data, distances, is_empty):
    #   o, n, q = data.shape
    #   m = o * q
    #   arange = torch.arange(m, device="cuda:0")
    #   data = data.transpose(0, 1).reshape(n,m)
    #   data = data[~is_empty ]
    #   result = distances[arange, :, data[:].long() ].sum(dim=1).t()
    #   return result

    return (final_v, final_i)