import torch
import numpy as np
from .BaseIndex import BaseIndex
from ..container import CellContainer
from ..codec import PQCodec, VQCodec
from ..kernels import IVFPQTopkCuda
from ..kernels import IVFPQTop1Cuda
from ..kernels import TopkBMMCuda
from ..kernels import MinBMMCuda
from .. import util

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
    self.distance = distance
    self.verbose = verbose
    self.pq_use_residual = pq_use_residual
    self.n_probe = 1

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

    self._top1024_cuda = IVFPQTopkCuda(
      m=n_subvectors,
      tpb=1024,
      n_cs=self.contiguous_size,
      sm_size=n_subvectors * 1024,
    )

    self._top512_cuda = IVFPQTopkCuda(
      m=n_subvectors,
      tpb=512,
      n_cs=self.contiguous_size,
      sm_size=n_subvectors * 1024,
    )

    self._top256_cuda = IVFPQTopkCuda(
      m=n_subvectors,
      tpb=512,
      n_cs=self.contiguous_size,
      sm_size=n_subvectors * 1024,
    )

    self._top1_cuda = IVFPQTop1Cuda(
      m=n_subvectors,
      tpb=512,
      n_cs=self.contiguous_size,
      sm_size=n_subvectors * 1024,
    )

    self._l2_min_cuda = MinBMMCuda(
      4, 4, "l2"
    )

    self._l2_topk_cuda = TopkBMMCuda(
      4, 4, "nl2"
    )

    self.warmup_kernels()

  def warmup_kernels(self):
    a = torch.randn(128, 128, device=self.device)
    b = torch.randn(128, 128, device=self.device)
    self._l2_min_cuda(a, b, dim=0)
    self._l2_topk_cuda(a, b, dim=0, k=128)

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
      x = pq_code, vq_code
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

  def search(self, x, k=1):
    assert len(x.shape) == 2
    assert x.shape[0] == self.d_vector
    assert 0 < k <= 1024
    if self.distance == "cosine":
      x = util.normalize(x, dim=0)
    n_query = x.shape[1]
    storage = self._storage
    is_empty = self._is_empty
    codebook = self.vq_codec.codebook
    if self.pq_use_residual:
      pass
      # precomputed = self.pq_codec.precompute_adc(x - recon)
    else:
      precomputed = self.pq_codec.precompute_adc(x)
    if self.n_probe == 1:
      topk_sims, topk_labels = self._l2_min_cuda(x.T, codebook, dim=1)
      topk_labels = topk_labels[:, None]
    else:
      _, topk_labels = self._l2_topk_cuda(x.T, codebook, k=self.n_probe, dim=1)
    cell_start = self._cell_start[topk_labels]
    cell_size = self._cell_size[topk_labels]
    if k == 1:
      topk_fn = self._top1_cuda
    elif k <= 256:
      topk_fn = self._top256_cuda
    elif k <= 512:
      topk_fn = self._top512_cuda
    elif k <= 1024:
      topk_fn = self._top1024_cuda

    topk_val, topk_address = topk_fn(
      data=self._storage,
      precomputed=precomputed,
      is_empty=self._is_empty,
      div_start=cell_start,
      div_size=cell_size,
      n_candidates = k,
    )
    topk_ids = self.get_id_by_address(topk_address)
    return topk_val, topk_ids