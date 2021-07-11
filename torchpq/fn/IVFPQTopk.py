from ..kernels import IVFPQTopkCuda
from ..kernels import IVFPQTop1Cuda

class IVFPQTopk:
  def __init__(
      self,
      n_subvectors,
      contiguous_size=4,
      sm_size=48*1024,
    ):
    self.n_subvectors = n_subvectors
    self.contiguous_size = contiguous_size
    self.sm_size = sm_size

    self._top1024_cuda = IVFPQTopkCuda(
      m=n_subvectors,
      tpb=1024,
      n_cs=contiguous_size,
      stack_capacity = 2,
      sm_size=n_subvectors * 1024,
    )

    self._top512_cuda = IVFPQTopkCuda(
      m=n_subvectors,
      tpb=512,
      n_cs=contiguous_size,
      stack_capacity = 2,
      sm_size=n_subvectors * 1024,
    )

    self._top256_cuda = IVFPQTopkCuda(
      m=n_subvectors,
      tpb=256,
      n_cs=contiguous_size,
      stack_capacity = 2,
      sm_size=n_subvectors * 1024,
    )

    if n_subvectors <= 32:
      self._top1_cuda = IVFPQTop1Cuda(
        m=n_subvectors,
        tpb=512,
        n_cs=contiguous_size,
        sm_size=n_subvectors * 1024,
      )
    else:
      self._top1_cuda = IVFPQTop1Cuda(
        m=n_subvectors,
        tpb=256,
        n_cs=contiguous_size,
        sm_size=n_subvectors * 1024,
      )

  def topk(
      self,
      data,
      precomputed,
      cell_start,
      cell_size,
      is_empty,
      n_probe_list=None,
      k=256
    ):
    assert 0 < k <= 1024
    if n_probe_list is not None:
      if k == 1:
        return self._top1_cuda.top1_smart_probing(
          data=data,
          precomputed=precomputed,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_probe_list=n_probe_list,
          n_candidates=k
        )
      elif 1 < k <= 256:
        return self._top256_cuda.topk_smart_probing(
          data=data,
          precomputed=precomputed,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_probe_list=n_probe_list,
          n_candidates=k
        )
      elif 256 < k <= 512:
        return self._top512_cuda.topk_smart_probing(
          data=data,
          precomputed=precomputed,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_probe_list=n_probe_list,
          n_candidates=k
        )
      elif 512 < k <= 1024:
        return self._top1024_cuda.topk_smart_probing(
          data=data,
          precomputed=precomputed,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_probe_list=n_probe_list,
          n_candidates=k
        )
    else:
      if k == 1:
        return self._top1_cuda.top1(
          data=data,
          precomputed=precomputed,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_candidates=k
        )
      elif 1 < k <= 256:
        return self._top256_cuda.topk(
          data=data,
          precomputed=precomputed,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_candidates=k
        )
      elif 256 < k <= 512:
        return self._top512_cuda.topk(
          data=data,
          precomputed=precomputed,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_candidates=k
        )
      elif 512 < k <= 1024:
        return self._top1024_cuda.topk(
          data=data,
          precomputed=precomputed,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_candidates=k
        )
  
  def topk_residual(
      self,
      data,
      precomputed,
      cell_start,
      cell_size,
      base_sims,
      is_empty,
      n_probe_list=None,
      k=256
    ):
    assert 0 < k <= 1024
    if n_probe_list is not None:
      if k == 1:
        return self._top1_cuda.top1_residual_smart_probing(
          data=data,
          precomputed=precomputed,
          base_sims=base_sims,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_probe_list=n_probe_list,
          n_candidates=k
        )
      elif 1 < k <= 256:
        return self._top256_cuda.topk_residual_smart_probing(
          data=data,
          precomputed=precomputed,
          base_sims=base_sims,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_probe_list=n_probe_list,
          n_candidates=k
        )
      elif 256 < k <= 512:
        return self._top512_cuda.topk_residual_smart_probing(
          data=data,
          precomputed=precomputed,
          base_sims=base_sims,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_probe_list=n_probe_list,
          n_candidates=k
        )
      elif 512 < k <= 1024:
        return self._top1024_cuda.topk_residual_smart_probing(
          data=data,
          precomputed=precomputed,
          base_sims=base_sims,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_probe_list=n_probe_list,
          n_candidates=k
        )
    else:
      if k == 1:
        return self._top1_cuda.top1_residual(
          data=data,
          precomputed=precomputed,
          base_sims=base_sims,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_candidates=k
        )
      elif 1 < k <= 256:
        return self._top256_cuda.topk_residual(
          data=data,
          precomputed=precomputed,
          base_sims=base_sims,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_candidates=k
        )
      elif 256 < k <= 512:
        return self._top512_cuda.topk_residual(
          data=data,
          precomputed=precomputed,
          base_sims=base_sims,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_candidates=k
        )
      elif 512 < k <= 1024:
        return self._top1024_cuda.topk_residual(
          data=data,
          precomputed=precomputed,
          base_sims=base_sims,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_candidates=k
        )

  def topk_residual_precomputed(
      self,
      data,
      part1,
      part2,
      cell_start,
      cell_size,
      cells,
      base_sims,
      is_empty,
      n_probe_list=None,
      k=256
    ):
    assert 0 < k <= 1024
    if n_probe_list is not None:
      if k == 1:
        return self._top1_cuda.top1_residual_precomputed_smart_probing(
          data=data,
          part1=part1,
          part2=part2,
          cells=cells,
          base_sims=base_sims,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_probe_list=n_probe_list,
          n_candidates=k
        )
      elif 1 < k <= 256:
        return self._top256_cuda.topk_residual_precomputed_smart_probing(
          data=data,
          part1=part1,
          part2=part2,
          cells=cells,
          base_sims=base_sims,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_probe_list=n_probe_list,
          n_candidates=k
        )
      elif 256 < k <= 512:
        return self._top512_cuda.topk_residual_precomputed_smart_probing(
          data=data,
          part1=part1,
          part2=part2,
          cells=cells,
          base_sims=base_sims,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_probe_list=n_probe_list,
          n_candidates=k
        )
      elif 512 < k <= 1024:
        return self._top1024_cuda.topk_residual_precomputed_smart_probing(
          data=data,
          part1=part1,
          part2=part2,
          cells=cells,
          base_sims=base_sims,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_probe_list=n_probe_list,
          n_candidates=k
        )
    else:
      if k == 1:
        return self._top1_cuda.top1_residual_precomputed(
          data=data,
          part1=part1,
          part2=part2,
          cells=cells,
          base_sims=base_sims,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_candidates=k
        )
      elif 1 < k <= 256:
        return self._top256_cuda.topk_residual_precomputed(
          data=data,
          part1=part1,
          part2=part2,
          cells=cells,
          base_sims=base_sims,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_candidates=k
        )
      elif 256 < k <= 512:
        return self._top512_cuda.topk_residual_precomputed(
          data=data,
          part1=part1,
          part2=part2,
          cells=cells,
          base_sims=base_sims,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_candidates=k
        )
      elif 512 < k <= 1024:
        return self._top1024_cuda.topk_residual_precomputed(
          data=data,
          part1=part1,
          part2=part2,
          cells=cells,
          base_sims=base_sims,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_candidates=k
        )
