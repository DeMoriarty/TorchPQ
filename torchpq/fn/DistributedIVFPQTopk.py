from ..kernels import DistributedIVFPQTopkCuda
from ..kernels import DistributedIVFPQTop1Cuda

class DistributedIVFPQTopk:
  def __init__(
      self,
      n_subvectors,
      contiguous_size=4,
      sm_size=48*1024,
    ):
    self.n_subvectors = n_subvectors
    self.contiguous_size = contiguous_size
    self.sm_size = sm_size

    self._top1024_cuda = DistributedIVFPQTopkCuda(
      m=n_subvectors,
      tpb=1024,
      n_cs=contiguous_size,
      stack_capacity = 2,
      sm_size=n_subvectors * 1024,
    )

    self._top512_cuda = DistributedIVFPQTopkCuda(
      m=n_subvectors,
      tpb=512,
      n_cs=contiguous_size,
      stack_capacity = 2,
      sm_size=n_subvectors * 1024,
    )

    self._top256_cuda = DistributedIVFPQTopkCuda(
      m=n_subvectors,
      tpb=256,
      n_cs=contiguous_size,
      stack_capacity = 2,
      sm_size=n_subvectors * 1024,
    )

    if n_subvectors <= 32:
      self._top1_cuda = DistributedIVFPQTop1Cuda(
        m=n_subvectors,
        tpb=512,
        n_cs=contiguous_size,
        sm_size=n_subvectors * 1024,
      )
    else:
      self._top1_cuda = DistributedIVFPQTop1Cuda(
        m=n_subvectors,
        tpb=256,
        n_cs=contiguous_size,
        sm_size=n_subvectors * 1024,
      )

  def topk(
      self,
      address2id_ptr,
      precomputed,
      cell_ptr,
      cell_size,
      cell_capacity,
      n_probe_list,
      k=256
    ):
    assert 0 < k <= 1024
    if k == 1:
      return self._top1_cuda.topk(
        address2id_ptr=address2id_ptr,
        precomputed=precomputed,
        cell_ptr=cell_ptr,
        cell_size=cell_size,
        cell_capacity=cell_capacity,
        n_probe_list=n_probe_list,
        n_candidates=k
      )
    elif 1 < k <= 256:
      return self._top256_cuda.topk(
        address2id_ptr=address2id_ptr,
        precomputed=precomputed,
        cell_ptr=cell_ptr,
        cell_size=cell_size,
        cell_capacity=cell_capacity,
        n_probe_list=n_probe_list,
        n_candidates=k
      )
    elif 256 < k <= 512:
      return self._top512_cuda.topk(
        address2id_ptr=address2id_ptr,
        precomputed=precomputed,
        cell_ptr=cell_ptr,
        cell_size=cell_size,
        cell_capacity=cell_capacity,
        n_probe_list=n_probe_list,
        n_candidates=k
      )
    elif 512 < k <= 1024:
      return self._top1024_cuda.topk(
        address2id_ptr=address2id_ptr,
        precomputed=precomputed,
        cell_ptr=cell_ptr,
        cell_size=cell_size,
        cell_capacity=cell_capacity,
        n_probe_list=n_probe_list,
        n_candidates=k
      )
  
  def topk_residual(
      self,
      address2id_ptr,
      precomputed,
      cell_ptr,
      cell_size,
      cell_capacity,
      base_sims,
      n_probe_list,
      k=256
    ):
    assert 0 < k <= 1024
    if k == 1:
      return self._top1_cuda.topk_residual(
        address2id_ptr=address2id_ptr,
        precomputed=precomputed,
        base_sims=base_sims,
        cell_ptr=cell_ptr,
        cell_size=cell_size,
        cell_capacity=cell_capacity,
        n_probe_list=n_probe_list,
        n_candidates=k
      )
    elif 1 < k <= 256:
      return self._top256_cuda.topk_residual(
        address2id_ptr=address2id_ptr,
        precomputed=precomputed,
        base_sims=base_sims,
        cell_ptr=cell_ptr,
        cell_size=cell_size,
        cell_capacity=cell_capacity,
        n_probe_list=n_probe_list,
        n_candidates=k
      )
    elif 256 < k <= 512:
      return self._top512_cuda.topk_residual(
        address2id_ptr=address2id_ptr,
        precomputed=precomputed,
        base_sims=base_sims,
        cell_ptr=cell_ptr,
        cell_size=cell_size,
        cell_capacity=cell_capacity,
        n_probe_list=n_probe_list,
        n_candidates=k
      )
    elif 512 < k <= 1024:
      return self._top1024_cuda.topk_residual(
        address2id_ptr=address2id_ptr,
        precomputed=precomputed,
        base_sims=base_sims,
        cell_ptr=cell_ptr,
        cell_size=cell_size,
        cell_capacity=cell_capacity,
        n_probe_list=n_probe_list,
        n_candidates=k
      )

  def topk_residual_precomputed(
      self,
      address2id_ptr,
      part1,
      part2,
      cell_ptr,
      cell_size,
      cell_capacity,
      cells,
      base_sims,
      n_probe_list=None,
      k=256
    ):
    assert 0 < k <= 1024
    if k == 1:
      return self._top1_cuda.topk_residual_precomputed(
        address2id_ptr=address2id_ptr,
        part1=part1,
        part2=part2,
        cells=cells,
        base_sims=base_sims,
        cell_ptr=cell_ptr,
        cell_size=cell_size,
        cell_capacity=cell_capacity,
        n_probe_list=n_probe_list,
        n_candidates=k
      )
    elif 1 < k <= 256:
      return self._top256_cuda.topk_residual_precomputed(
        address2id_ptr=address2id_ptr,
        part1=part1,
        part2=part2,
        cells=cells,
        base_sims=base_sims,
        cell_ptr=cell_ptr,
        cell_size=cell_size,
        cell_capacity=cell_capacity,
        n_probe_list=n_probe_list,
        n_candidates=k
      )
    elif 256 < k <= 512:
      return self._top512_cuda.topk_residual_precomputed(
        address2id_ptr=address2id_ptr,
        part1=part1,
        part2=part2,
        cells=cells,
        base_sims=base_sims,
        cell_ptr=cell_ptr,
        cell_size=cell_size,
        cell_capacity=cell_capacity,
        n_probe_list=n_probe_list,
        n_candidates=k
      )
    elif 512 < k <= 1024:
      return self._top1024_cuda.topk_residual_precomputed(
        address2id_ptr=address2id_ptr,
        part1=part1,
        part2=part2,
        cells=cells,
        base_sims=base_sims,
        cell_ptr=cell_ptr,
        cell_size=cell_size,
        cell_capacity=cell_capacity,
        n_probe_list=n_probe_list,
        n_candidates=k
      )
