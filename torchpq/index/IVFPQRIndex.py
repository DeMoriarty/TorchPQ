from .BaseIndex import BaseIndex

class IVFPQRIndex(CellContainer):
  def __init__(
      self,
      d_vector,
      n_subvectors=8,
      n_subvectors_rerank=8,
      n_cells=128,
      use_residual=True,
      initial_size=None,
      expand_step_size=128,
      expand_mode="double",
      distance="euclidean",
      device='cuda:0',
      verbose=0,
    ):
    if torch.device(device).type == "cuda":
      assert torch.cuda.is_available(), "cuda is not available"
      max_sm_bytes = util.get_maximum_shared_memory_bytes()
      assert n_subvectors <= max_sm_bytes // 1024
      assert n_subvectors_rerank <= max_sm_bytes // 1024

    super(IVFPQRIndex, self).__init__(
      code_size = n_subvectors + n_subvectors_rerank,
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
    self.n_subvectors_rerank = n_subvectors_rerank
    self.use_residual = use_residual
    self.verbose = verbose

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

    self.pq_rerank_codec = PQCodec(
      d_vector = d_vector,
      n_subvectors = n_subvectors_rerank,
      n_clusters = 256,
      distance = distance,
      verbose = verbose
    )

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

  def set_pq_rerank_codec_max_iter(self, value):
    self.pq_rerank_codec.kmeans.max_iter = value

  def set_pq_rerank_codec_n_redo(self, value):
    self.pq_rerank_codec.kmeans.n_redo = value

  def set_pq_rerank_codec_tolerance(self, value):
    self.pq_rerank_codec.kmeans.tol = value

  def train(self, x, force_retrain = False):
    if self.vq_codec.is_trained and self.pq_codec.is_trained:
      if not force_retrain:
        self.print_message("index is already trained")
        return
    assert len(x.shape) == 2
    assert x.shape[0] == self.d_vector
    if self.distance == "cosine":
      x = util.normalize(x, dim=0)
    d_vector, n_data = x.shape

    self.print_message("start training VQ codec...")
    self.vq_codec.train(x)

    self.print_message("start training PQ codec...")
    code = self.pq_codec.train(x)

    if self.use_residual:
      y = self.pq_codec.decode(code)
      x = x - y
      self.print_message("start training PQ Rerank codec with residuals...")
    else:
      self.print_message("start training PQ Rerank codec...")
    self.pq_codec.train(x)

    self.print_message("index is trained successfully!")
  
  def encode(self, x):
    """
      Encode `x` with PQ and PQ Rerank codecs
      x:
        torch.Tensor
        dtype : float32
        shape : [d_vector, n_data]

      returns:
        torch.Tensor
        dtype : uint8
        shape : [n_subvectors + n_subvectors_rerank, n_data]
    """
    assert len(x.shape) == 2
    assert x.shape[0] == self.d_vector
    if self.distance == "cosine":
      x = util.normalize(x)
    y1 = self.pq_codec.encode(x)
    if self.use_residual:
      recon = self.pq_codec.decode(y1)
      x = x - recon
    y2 = self.pq_rerank_codec.encode(x)
    y = torch.cat([y1, y2], dim=0)
    return y

  def decode(self, x):
     """
      Decode `x` with PQ and PQ Rerank codecs
      x:
        torch.Tensor
        dtype : uint8
        shape : [n_subvectors + n_subvectors_rerank, n_data]

      returns:
        torch.Tensor
        dtype : float32
        shape : [d_vector, n_data]
    """
    assert len(x.shape) == 2
    assert x.shape[0] == self.n_subvectors + self.n_subvectors_rerank

    x1 = x[:self.n_subvectors, :] #[n_subvectors, n_data]
    x2 = x[self.n_subvectors:, :] #[n_subvectors_r, n_data]
    y = self.pq_rerank_codec.decode(x2)
    if self.use_residual:
      residual = self.pq_codec.decode(x1)
      y = y + residual
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

    assigned_cells = self.vq_codec.predict(x)
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
    if self.distance == "cosine":
      x = util.normalize(x, dim=0)
    raise NotImplementedError