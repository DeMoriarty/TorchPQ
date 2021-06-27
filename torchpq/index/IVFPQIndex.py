import torch
import numpy as np
from .BaseIndex import BaseIndex
from ..container import CellContainer
from ..codec import PQCodec, VQCodec
from ..kernels import IVFPQTopkCuda
from ..kernels import IVFPQTop1Cuda
from .. import util
from .. import metric

class IVFPQIndex(CellContainer):
  def __init__(
      self,
      d_vector,
      n_subvectors=8,
      n_cells=128,
      initial_size=None,
      expand_step_size=128,
      expand_mode="double",
      distance="euclidean",
      device='cuda:0',
      pq_use_residual=False,
      verbose=0,
    ):
    if torch.device(device).type == "cuda":
      assert torch.cuda.is_available(), "cuda is not available"
      max_sm_bytes = util.get_maximum_shared_memory_bytes()
      assert n_subvectors <= max_sm_bytes // 1024
    assert d_vector % n_subvectors == 0

    super(IVFPQIndex, self).__init__(
      code_size = n_subvectors,
      n_cells = n_cells,
      dtype = "uint8",
      device = device,
      initial_size = initial_size,
      expand_step_size = expand_step_size,
      expand_mode = expand_mode,
      use_inverse_id_mapping = True,
      contiguous_size = 4,
      verbose = verbose,
    )

    self.d_vector = d_vector
    self.n_subvectors = n_subvectors
    self.d_subvector = d_vector // n_subvectors
    self.distance = distance
    self.verbose = verbose
    self.pq_use_residual = pq_use_residual
    self.n_probe = 1
    if pq_use_residual and (n_cells * 256 * n_subvectors * 4) <= 4*1024**3:
      self._use_precomputed = True
    else:
      self._use_precomputed = False
    self._precomputed_part2 = None
    self._use_cublas = False
    self._use_smart_probing = True
    self._smart_probing_temperature = 30.0

    self.vq_codec = VQCodec(
      n_clusters = n_cells,
      n_redo = 1,
      max_iter = 15,
      tol = 1e-4,
      distance = "euclidean",
      init_mode = "random",
      verbose = verbose
    )

    self.pq_codec = PQCodec(
      d_vector = d_vector,
      n_subvectors = n_subvectors,
      n_clusters = 256,
      distance = distance,
      verbose = verbose
    )

    self._ivfpq_topk = IVFPQTopk(
      n_subvectors = n_subvectors,
      contiguous_size = self.contiguous_size,
      sm_size = n_subvectors * 1024,
    )

  @property
  def use_cublas(self):
    return self._use_cublas

  @use_cublas.setter
  def use_cublas(self, value):
    assert type(value) is bool
    self._use_cublas = value

  @property
  def use_smart_probing(self):
    return self._use_smart_probing

  @use_smart_probing.setter
  def use_smart_probing(self, value):
    assert type(value) is bool
    self._use_smart_probing = value

  @property
  def smart_probing_temperature(self):
    return self._smart_probing_temperature

  @smart_probing_temperature.setter
  def smart_probing_temperature(self, value):
    assert value > 0
    assert self.use_smart_probing, "set use_smart_probing to True first"
    self._smart_probing_temperature = value

  @property
  def use_precomputed(self):
    return self._use_precomputed

  @use_precomputed.setter
  def use_precomputed(self, value):
    assert type(value) is bool
    if value:
      assert self.pq_use_residual, " `use_precomputed=True` is only valid when `pq_use_residual` is True"
      assert (self.pq_codec.is_trained and self.vq_codec.is_trained), "index is not trained"
      required_memory = (self.n_cells * self.n_subvectors * 1024) / 1024**2
      self.print_message(f"estimated memory for precomputed table is {required_memory} MB", 1)
      self.precompute_part2()
    else:
      self._precomputed_part2 = None
    self._use_precomputed = value

  def precompute_part2(self):
    pq_codebook = self.pq_codec.codebook
    vq_codebook = self.vq_codec.codebook.reshape(
      self.n_subvectors,
      self.d_subvector,
      self.n_cells
    )
    self._precomputed_part2 = torch.bmm(
      vq_codebook.transpose(-1, -2),
      pq_codebook
    ) * -2 - pq_codebook.norm(dim=1).pow(2)[:, None]

  @property
  def vq_codec_max_iter(self):
    return self.vq_codec.kmeans.max_iter

  @vq_codec_max_iter.setter
  def vq_codec_max_iter(self, value):
    assert type(value) is int
    assert value > 0
    assert not self.vq_codec.is_trained, "vq_codec is already trained"
    self.vq_codec.kmeans.max_iter = value

  @property
  def vq_codec_n_redo(self):
    return self.vq_codec.kmeans.n_redo

  @vq_codec_n_redo.setter
  def vq_codec_n_redo(self, value):
    assert type(value) is int
    assert value > 0
    assert not self.vq_codec.is_trained, "vq_codec is already trained"
    self.vq_codec.kmeans.n_redo = value
  
  @property
  def vq_codec_tolerance(self):
    return self.vq_codec.kmeans.tol

  @vq_codec_tolerance.setter
  def vq_codec_tolerance(self, value):
    assert not self.vq_codec.is_trained, "vq_codec is already trained"
    self.vq_codec.kmeans.tol = value

  @property
  def pq_codec_max_iter(self):
    return self.pq_codec.kmeans.max_iter
  
  @pq_codec_max_iter.setter
  def pq_codec_max_iter(self, value):
    assert type(value) is int
    assert value > 0
    assert not self.pq_codec.is_trained, "pq_codec is already trained"
    self.pq_codec.kmeans.max_iter = value

  @property
  def pq_codec_n_redo(self):
    return self.pq_codec.kmeans.n_redo

  @pq_codec_n_redo.setter
  def pq_codec_n_redo(self, value):
    assert type(value) is int
    assert value > 0
    assert not self.pq_codec.is_trained, "pq_codec is already trained"
    self.pq_codec.kmeans.n_redo = value

  @property
  def pq_codec_tolerance(self):
    return self.pq_codec.kmeans.tol
  
  @pq_codec_tolerance.setter
  def pq_codec_tolerance(self, value):
    assert not self.pq_codec.is_trained, "pq_codec is already trained"
    self.pq_codec.kmeans.tol = value

  def train(self, x, force_retrain = False):
    if self.vq_codec.is_trained and self.pq_codec.is_trained:
      if not force_retrain:
        self.print_message("index is already trained", 1)
        return
    assert len(x.shape) == 2
    assert x.shape[0] == self.d_vector
    if self.distance == "cosine":
      x = util.normalize(x, dim=0)
    d_vector, n_data = x.shape

    self.print_message("start training VQ codec...", 1)
    code = self.vq_codec.train(x)

    self.print_message("start training PQ codec...", 1)
    if self.pq_use_residual:
      recon = self.vq_codec.decode(code)
      self.pq_codec.train(x - recon)
    else:
      self.pq_codec.train(x)

    self.print_message("index is trained successfully!", 1)
  
  def encode(self, x):
    """
      Encode `x` with PQ codec
      x:
        torch.Tensor
        dtype : float32
        shape : [d_vector, n_data]

      returns:
        torch.Tensor
        dtype : uint8
        shape : [n_subvectors, n_data]
    """
    assert len(x.shape) == 2
    assert x.shape[0] == self.d_vector
    if self.distance == "cosine":
      x = util.normalize(x)
    if self.pq_use_residual:
      vq_code = self.vq_codec.encode(x)
      recon = self.vq_codec.decode(vq_code)
      pq_code = self.pq_codec.encode(x - recon)
      return pq_code, vq_code
    else:
      y = self.pq_codec.encode(x)
      return y

  def decode(self, x):
    """
      Decode `x` with PQ codec
      x:
        torch.Tensor
        dtype : uint8
        shape : [n_subvectors, n_data]

      returns:
        torch.Tensor
        dtype : float32
        shape : [d_vector, n_data]
    """
    if self.pq_use_residual:
      assert len(x) == 2
      pq_code, vq_code = x
      assert pq_code.shape[0] == self.n_subvectors
      assert pq_code.shape[1] == vq_code.shape[0]
      residual = self.pq_codec.decode(pq_code)
      recon = self.vq_codec.decode(vq_code)
      y = recon + residual

    else:
      assert len(x.shape) == 2
      assert x.shape[0] == self.n_subvectors
      y = self.pq_codec.decode(x)
    return y
  
  def add(self, x, ids=None, return_address=False):
    """
      Add `x` to index, with optional `ids` for each vector in `x`
      x:
        torch.Tensor
        dtype : float32
        shape : [d_vector, n_data]

      ids: optional
        torch.Tensor
        dtype : int64
        shape : [n_data]
        If not given, or given None, `ids` will be set to 
        torch.arange(n_data) + self.max_id + 1

      return_address:
        bool
        default : False
        if set to True, return address of the added vectors

      returns (ids) or (ids, address):
        ids:
          torch.Tensor
          dtype : int64
          shape : [n_data]
        
        address:
          torch.Tensor
          dtype : int64
          shape : [n_data]
          this is returned if `return_address` is True
    """
    assert len(x.shape) == 2
    assert x.shape[0] == self.d_vector
    if self.distance == "cosine":
      x = util.normalize(x)

    assigned_cells = self.vq_codec.encode(x)
    if self.pq_use_residual:
      quantized_x, _ = self.encode(x)
    else:
      quantized_x = self.encode(x)

    return super(IVFPQIndex, self).add(
      quantized_x,
      cells=assigned_cells,
      ids=ids,
      return_address = return_address
    )

  def precomputed_adc_residual_precomputed(self, x):
    n_query = x.shape[1]
    vq_codebook = self.vq_codec.codebook #[d_vector, n_cells]
    pq_codebook = self.pq_codec.codebook #[n_subvectors, d_subvectors, 256]
    x = x.reshape(
      self.n_subvectors,
      self.d_subvector,
      n_query
    ).transpose(-1, -2)
    part1 = 2 * (x @ pq_codebook).permute(1, 0, 2) #[n_query, n_sub, 256]
    if self._precomputed_part2 is None:
      self.precompute_part2()
    part2 = self._precomputed_part2.transpose(0, 1)
    return part1, part2

  def precomputed_adc_residual(self, x, cells):
    # shape of cells: [n_query, n_probe]
    # shape of x: [d_vector, n_query]
    n_query = x.shape[1]
    n_probe = cells.shape[1]
    vq_codebook = self.vq_codec.codebook #[d_vector, n_cells]
    pq_codebook = self.pq_codec.codebook #[n_subvectors, d_subvectors, 256]
    x = x.reshape(
      self.n_subvectors,
      self.d_subvector,
      n_query
    ).transpose(-1, -2)
    part1 = 2 * (x @ pq_codebook).permute(1, 0, 2) - pq_codebook.norm(dim=1).pow(2)[None]#[n_query, n_sub, 256]

    vq_codebook = vq_codebook.reshape(
      self.n_subvectors,
      self.d_subvector,
      self.n_cells
    )
    vq_codebook = vq_codebook[:, :, cells]#[n_sub, d_sub, n_query, n_probe]
    vq_codebook = vq_codebook.permute(0, 2, 3, 1)#[n_sub, n_query, n_probe, d_sub]
    pq_codebook = pq_codebook[:, None].expand(-1, n_query, -1, -1)#[n_sub, n_query, d_sub, 256]
    part2 = - 2 * (vq_codebook @ pq_codebook)#[n_sub, n_query, n_probe, 256]
    precomputed = part1[:, None] + part2.permute(1, 2, 0, 3)#[n_query, n_probe, n_sub, 256]
    return precomputed

  def search_cells(
      self,
      x,
      cells,
      base_sims=None,
      n_probe_list=None,
      k=1,
      return_address=False
    ):
    storage = self._storage
    is_empty = self._is_empty
    vq_codebook = self.vq_codec.codebook

    cell_start = self._cell_start[cells]
    cell_size = self._cell_size[cells]
    
    if self.pq_use_residual:
      assert base_sims is not None, "base_sims is required when pq_use_residual is True"
      if self.use_precomputed:
        part1, part2 = self.precomputed_adc_residual_precomputed(x)
        topk_val, topk_address = self._ivfpq_topk.topk_residual_precomputed(
          data=storage,
          part1=part1,
          part2=part2,
          cells=cells,
          base_sims=base_sims,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_probe_list=n_probe_list,
          k=k
        )
      else:
        precomputed = self.precomputed_adc_residual(x, cells)
        topk_val, topk_address = self._ivfpq_topk.topk_residual(
          data=storage,
          base_sims=base_sims,
          precomputed=precomputed,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          n_probe_list=n_probe_list,
          k=k
        )
    else:
      precomputed = self.pq_codec.precompute_adc(x)
      topk_val, topk_address = self._ivfpq_topk.topk(
        data=storage,
        precomputed=precomputed,
        cell_start=cell_start,
        cell_size=cell_size,
        is_empty=is_empty,
        n_probe_list=n_probe_list,
        k=k
      )

    topk_ids = self.get_id_by_address(topk_address)
    if return_address:
      return topk_val, topk_ids, topk_address
    else:
      return topk_val, topk_ids

  def search(self, x, k=1, return_address=False):
    assert len(x.shape) == 2
    assert x.shape[0] == self.d_vector
    assert 0 < k <= 1024
    if self.distance == "cosine":
      x = util.normalize(x, dim=0)
    n_query = x.shape[1]
    storage = self._storage
    is_empty = self._is_empty
    vq_codebook = self.vq_codec.codebook

    # find n_probe closest cells
    if self.use_cublas:
      sims = metric.negative_squared_l2_distance(x, vq_codebook)
      topk_sims, cells = sims.topk(k=self.n_probe, dim=1)
    else:
      topk_sims, cells = self.vq_codec.kmeans.topk(x, k=self.n_probe)
    
    if self.use_smart_probing and self.n_probe > 1:
      p = -topk_sims.abs().sqrt()
      p = torch.softmax(p / self.temperature, dim=-1)
      p_norm = p.norm(dim=-1)
      sqrt_d = self.n_probe ** 0.5
      score = 1 - (p_norm * sqrt_d - 1) / (sqrt_d - 1) - 1e-6
      n_probe_list = torch.ceil(score * (self.n_probe) ).long()
      # print("average new n probe", new_n_probe.float().mean() / self.n_probe)
    else:
      n_probe_list = None

    return self.search_cells(
      x=x, 
      cells=cells,
      base_sims=topk_sims,
      n_probe_list=n_probe_list,
      k=k,
      return_address=False
    )
    
    


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
      sm_size=n_subvectors * 1024,
    )

    self._top512_cuda = IVFPQTopkCuda(
      m=n_subvectors,
      tpb=512,
      n_cs=contiguous_size,
      sm_size=n_subvectors * 1024,
    )

    self._top256_cuda = IVFPQTopkCuda(
      m=n_subvectors,
      tpb=256,
      n_cs=contiguous_size,
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

  