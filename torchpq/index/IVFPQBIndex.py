import torch
import numpy as np
from ..container import CellContainer, FlatContainerGroup
from ..codec import PQCodec, VQCodec
from ..kernels import IVFPQTopkCuda
from ..kernels import IVFPQTop1Cuda
from .. import util
from .. import metric

class IVFPQBIndex(CellContainer):
  """
    Diplomatic Inverted File Product Quantization
  """
  def __init__(
      self,
      d_vector,
      n_subvectors=8,
      n_cells=128,
      n_neighbors = 16,
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
    assert 0 < n_neighbors <= n_cells

    super(IVFPQBIndex, self).__init__(
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
    self.n_neighbors = n_neighbors
    self.d_subvector = d_vector // n_subvectors
    self.distance = distance
    self.verbose = verbose
    
    self.n_probe_coarse = 32
    self.n_probe = 4
    self.pq_use_residual = pq_use_residual
    if pq_use_residual and (n_cells * 256 * n_subvectors * 4) <= 4*1024**3:
      self._use_precomputed = True
    else:
      self._use_precomputed = False
    self._precomputed_part2 = None
    self._use_cublas = False

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

    neighboring_cells = torch.zeros(
      n_cells,
      n_neighbors,
      dtype=torch.long,
      device=self.device,
    )
    self.register_buffer("_neighboring_cells", neighboring_cells)

    self._border = CellContainer(
      code_size = self.code_size,
      n_cells = n_cells,
      dtype = "uint8",
      device = device,
      initial_size = n_neighbors,
      expand_step_size = 1,
      expand_mode = "step",
      use_inverse_id_mapping = False,
      contiguous_size = 4,
      verbose = verbose,
    )
    self._border._is_empty.fill_(0)
    self._border._cell_size.fill_(n_neighbors)

    border_sims = torch.empty(
      n_cells,
      n_neighbors,
      dtype=torch.float32,
      device=self.device
    )
    border_sims.fill_(float("-inf"))
    self.register_buffer("_border_sims", border_sims)

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
    self._use_cublas = value

  @property
  def use_precomputed(self):
    return self._use_precomputed

  @use_precomputed.setter
  def use_precomputed(self, value):
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

  def set_vq_codec_max_iter(self, value):
    self.vq_codec.kmeans.max_iter = value

  def set_vq_codec_n_redo(self, value):
    self.vq_codec.kmeans.n_redo = value
  
  def set_vq_codec_tolerance(self, value):
    self.vq_codec.kmeans.tol = value

  def set_pq_codec_max_iter(self, value):
    self.pq_codec.kmeans.max_iter = value

  def set_pq_codec_n_redo(self, value):
    self.pq_codec.kmeans.n_redo = value

  def set_pq_codec_tolerance(self, value):
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

    self.print_message("searching neighbors...", 1)
    vq_codebook = self.vq_codec.codebook #[d_vector, n_cells]

    # _, topk_neighbors = self.vq_codec.kmeans.topk(
    #   vq_codebook,
    #   k=self.n_neighbors
    # ) #TODO: TopkBMMCuda bugged
    sims = self.vq_codec.kmeans.sim(
      vq_codebook,
      vq_codebook,
    )
    _, topk_neighbors = sims.topk(k=self.n_neighbors, dim=-1)

     #[n_cells, n_neighbors]
    self._neighboring_cells[:] = topk_neighbors
    quantized_vq_codebook = self.encode(vq_codebook) #[code_size, n_cells]
    address = torch.arange(
      self.n_cells,
      device=self.device
    ) * self.n_neighbors

    self._border.set_data_by_address(
      data = quantized_vq_codebook,
      address = address
    )

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
    # topk_sims, topk_cells = self.vq_codec.kmeans.topk(x, k=self.n_neighbors)
    # assigned_cells = topk_cells[:, 0]
    
    if self.pq_use_residual:
      quantized_x, _ = self.encode(x)
    else:
      quantized_x = self.encode(x)

    returned = super(IVFPQBIndex, self).add(
      quantized_x,
      cells=assigned_cells,
      ids=ids,
      return_address = return_address
    )

    vq_codebook = self.vq_codec.codebook
    unique_cells = assigned_cells.unique()
    for cell in unique_cells:
      mask = assigned_cells == cell
      neighbors = self._neighboring_cells[cell, 1:] #[n_neighbors-1]
      neighbor_centroids = vq_codebook[:, neighbors]
      selected_x = x[:, mask] #[d_vector, n_selected]
      sims = self.vq_codec.kmeans.sim(selected_x, neighbor_centroids) #[n_selected, n_neighbors-1]
      max_sim, max_sim_idx = sims.max(dim=0) #[n_neighbors-1]
      previous_border_sims = self._border_sims[cell, 1:] #[n_neighbor-1]
      final_border_sim, update_mask = torch.stack(
        (previous_border_sims, max_sim)
      ).max(dim=0) #[n_neighbors-1]
      new_border_idx = max_sim_idx[update_mask.bool()]
      self._border_sims[cell, 1:] = final_border_sim
      address = torch.arange(
        1,
        self.n_neighbors,
        device=self.device
      ) + cell * self.n_neighbors
      new_data = quantized_x[:, mask][:, new_border_idx]
      self._border.set_data_by_address(
        data = new_data,
        address = address
      )

    return returned

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

  def search(self, x, k=1):
    assert len(x.shape) == 2
    assert x.shape[0] == self.d_vector
    assert 0 < k <= 1024
    if self.distance == "cosine":
      x = util.normalize(x, dim=0)
    n_query = x.shape[1]
    storage = self._storage
    is_empty = self._is_empty
    vq_codebook = self.vq_codec.codebook

    # find n_probe_coarse closest cells
    if self.use_cublas:
      sims = metric.negative_squared_l2_distance(x, vq_codebook)
      topk_sims, coarse_cells = sims.topk(
        k=self.n_probe_coarse,
        dim=1
      ) #[n_query, n_probe_coarse]
    else:
      topk_sims, coarse_cells = self.vq_codec.kmeans.topk(
        x,
        k=self.n_probe_coarse
      ) #[n_query, n_probe_coarse]

    # cells, sorted_cell_indices = torch.sort(cells, dim=-1)
    # topk_sims = topk_sims[sorted_cell_indices]

    # cell_start = self._cell_start[cells]
    # cell_size = self._cell_size[cells]
    
    if self.pq_use_residual:
      if self.use_precomputed:
        part1, part2 = self.precomputed_adc_residual_precomputed(x)
        topk_val, topk_address = self._ivfpq_topk.topk_residual_precomputed(
          data=storage,
          part1=part1,
          part2=part2,
          cells=cells,
          base_sims=topk_sims,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          k=k
        )
      else:
        precomputed = self.precomputed_adc_residual(x, cells)
        topk_val, topk_address = self._ivfpq_topk.topk_residual(
          data=storage,
          base_sims=topk_sims,
          precomputed=precomputed,
          cell_start=cell_start,
          cell_size=cell_size,
          is_empty=is_empty,
          k=k
        )
    else:
      precomputed = self.pq_codec.precompute_adc(x)
      # Search in borders first
      _, topk_border_address = self._ivfpq_topk.topk(
        data=self._border._storage,
        precomputed=precomputed,
        cell_start=self._border._cell_start[coarse_cells],
        cell_size=self._border._cell_size[coarse_cells],
        is_empty=self._border._is_empty,
        k=256
      )
      topk_neighbors = topk_border_address // self.n_neighbors
      # topk_neighbors = self._border.get_cell_by_address(topk_border_address)
      
      # topk_neighbors = [i.unique()[:self.n_probe] for i in topk_neighbors]
      # def pad(x):
      #   y = torch.zeros(
      #     self.n_probe,
      #     device=x.device, 
      #     dtype=x.dtype
      #   ) - 1
      #   y[:x.shape[0]] = x
      #   return x
      # topk_neighbors = [i if i.shape[0] == self.n_probe else pad(i) for i in topk_neighbors]
      # topk_neighbors = torch.stack(topk_neighbors)
      topk_neighbors = topk_neighbors[:, :self.n_probe]
      topk_neighbors, _ = topk_neighbors.sort(dim=-1)

      cell_start = self._cell_start[topk_neighbors]
      cell_size = self._cell_size[topk_neighbors]
      # cell_size[topk_neighbors < 0] = 0

      topk_val, topk_address = self._ivfpq_topk.topk(
        data=storage,
        precomputed=precomputed,
        cell_start=cell_start,
        cell_size=cell_size,
        is_empty=is_empty,
        k=k
      )

    topk_ids = self.get_id_by_address(topk_address)
    return topk_val, topk_ids


class IVFPQTopk:
  def __init__(self, n_subvectors, contiguous_size=4, sm_size=48*1024):
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
      k=256
    ):
    assert 0 < k <= 1024
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
      k=256
    ):
    assert 0 < k <= 1024
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
      k=256
    ):
    assert 0 < k <= 1024
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

  