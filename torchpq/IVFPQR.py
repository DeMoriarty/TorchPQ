import torch
import numpy as np

from .IVFPQBase import IVFPQBase
from .kmeans import KMeans
from .PQ import PQ
from .IVFPQTopk import IVFPQTopk

class IVFPQR(IVFPQBase):
  def __init__(
    self,
    d_vector,
    n_subvectors=8,
    n_subvectors_r=8,
    n_cq_clusters=128,
    n_pq_clusters=256,
    blocksize=64,
    verbose=0,
    use_residual=True,
    rerank_factor=2,
    distance="euclidean",
    device='cuda:0'
    ):
    """
    this is an efficient implementation of IVFPQ+R algorithm
    for more details, you can search 'Searching in one billion vectors: re-rank with source coding'

    Parameters:
      d_vector: int
        dimentionality of vectors to be quantized.

      n_subvectors: int, default : 8
        number of sub-quantizers, needs to be a multiple of 4

      n_subvectors_r: int, default : 8
        number of sub-quantizers for PQ-rerank, need to be a multiple of 4
        maximum number of n_subvectors and n_subvectors_r depends on the GPU architecture
          GPU Architecture   fp32  fp16 
          Ampere (GA100)      163   326  
          Turing (TU102 etc.) 64    128  
          Volta (GV100)       96    192  
          Pascal and before   48    96   
      
      n_cq_clusters: int, default : 128
        number coarse quantizer clusters
        recommended value is between 4*sqrt(n_data) ~ 16*sqrt(n_data)

      n_pq_clusters: int, default : 256
        number of product quantizer clusters
        any value other than 256 is not recommended.

      blocksize: int, default : 64
        number of vectors that can be asigned to each cluster of coarse_quantizer initially
        can be expanded using .expand method, .add method will automatically call .expand if necessary
        recommended value is (n_data_to_be_stored / n_cq_clusters)

      verbose: int, default : 0
        verbosity

      distance: str, default : 'euclidean'
        type of distance metric
        can be one of: ['euclidean', 'cosine', 'inner']

      device: str, default : 'cuda:0'
        which device to use
        can be one of: ['cpu', 'cuda', 'cuda:X'] *X represents cuda device index
      
    """
    n_cs = 4
    self.n_cs = n_cs
    self.d_vector = d_vector
    self.d_subvector = d_vector // n_subvectors
    self.d_subvector_r = d_vector // n_subvectors_r
    assert n_subvectors % self.n_cs == 0, f"n_subvectors needs to be a multiple of {n_cs}"
    assert n_subvectors_r % self.n_cs == 0, f"n_subvectors_r needs to be a multiple of {n_cs}"
    self.n_subvectors = n_subvectors
    self.n_subvectors_r = n_subvectors_r
    self.n_pq_clusters = n_pq_clusters
    self.use_residual = use_residual
    self.rerank_factor = rerank_factor
    self.n_probe = 1

    cc = self.get_cc()
    if cc[0] < 7 or cc == (7, 2):
      assert n_subvectors <= 48
      assert n_subvectors_r <= 48
    elif cc == (7, 0):
      assert n_subvectors <= 96
      assert n_subvectors_r <= 96
    elif cc == (7, 5):
      assert n_subvectors <= 64
      assert n_subvectors_r <= 64
    elif cc == (8, 0):
      assert n_subvectors <= 163
      assert n_subvectors_r <= 163
    elif cc == (8, 6):
      assert n_subvectors <= 99
      assert n_subvectors_r <= 99
    
    super(IVFPQR, self).__init__(
      code_size=n_subvectors+n_subvectors_r,
      n_cq_clusters=n_cq_clusters,
      blocksize=blocksize,
      verbose=verbose,
      distance=distance,
      device=device
    )

    self.coarse_q = KMeans(
      max_iter=25,
      n_clusters=n_cq_clusters,
      distance="inner" if distance == "cosine" else distance,
      init_mode="random",
      verbose=verbose,
    )

    self.product_q = PQ(
      d_vector=d_vector,
      n_subvectors=n_subvectors,
      n_clusters=n_pq_clusters,
      distance="inner" if distance == "cosine" else distance,
      verbose=verbose,
    )
    
    self.product_q_r = PQ(
      d_vector=d_vector,
      n_subvectors=n_subvectors_r,
      n_clusters=n_pq_clusters,
      distance="inner" if distance == "cosine" else distance,
      verbose=verbose,
    )
    self._topk_fn = IVFPQTopk(
      n_subvectors=n_subvectors,
      n_clusters=n_pq_clusters,
      n_cs=n_cs
    )
    self._topk_r_fn = IVFPQTopk(
      n_subvectors=n_subvectors_r,
      n_clusters=n_pq_clusters,
      n_cs=n_cs
    )
  
  def set_coarse_q_max_iter(self, value):
    self.coarse_q.max_iter = value
  
  def set_product_q_max_iter(self, value):
    self.product_q.kmeans.max_iter = value

  def set_product_q_r_max_iter(self, value):
    self.product_q_r.kmeans.max_iter = value
  
  def set_coarse_q_n_redo(self, value):
    self.coarse_q.n_redo = value
  
  def set_product_q_n_redo(self, value):
    self.product_q.kmeans.n_redo = value

  def set_product_q_r_n_redo(self, value):
    self.product_q_r.kmeans.n_redo = value

  def get_data_of_address(self, address):
    """
      address: torch.Tensor, shape : [n_address], dtype : int64
      returns: torch.Tensor, shape : [(n_subvectors+n_subvectors_r), n_address], dtype : uint8
      address must be in range [0, tot_capacity)
    """
    assert address.dtype == torch.long
    n_address = address.shape[0]
    mask = address < 0
    data = self.storage[:, address] #[(n_subvectors+n_subvectors_r)//n_cs, n_address, n_cs]
    data = data.transpose(1,2).reshape(self.n_subvectors + self.n_subvectors_r, n_address)
    if mask.sum() > 0:
      data[:, mask] = 0
    return data
  
  def get_data_of_id(self, ids):
    """
      ids: torch.Tensor, shape : [n_ids], dtype : int64
      returns: torch.Tensor, [(n_subvectors+n_subvectors_r), n_ids]
      if an id doesn't exist, its retrieved data will be zeros.
    """
    address = self.get_address_of_id(ids)
    data = self.get_data_of_address(address)
    return data

  def set_data_of_address(self, data, address):
    """
      data: torch.Tensor, shape : [(n_subvectors+n_subvectors_r) // n_cs, n_data, n_cs], dtype : uint8
      address: torch.Tensor, shape : [n_data], dtype : int64
    """
    self.storage[:, address] = data

  def set_data_of_id(self, data, ids):
    """
      data: torch.Tensor, shape : [(n_subvectors, n_subvectors_r) // n_cs, n_data, n_cs], dtype : uint8
      ids: torch.Tensor, shape : [n_data], dtype : int64
    """
    address = self.get_address_of_id(ids)
    set_data_of_address(data, address)

  def add(self, input, input_ids=None):
    """
      input: torch.Tensor, shape : [n_data, d_vector], dtype : float32
      input_ids: torch.Tensor, shape : [n_data], dtype : int64
    """
    assert self._is_trained == True, "Module is not trained"
    if self.distance == "cosine":
      input = self.normalize(input)
    input = input.contiguous()

    d_vector, n_data = input.shape
    assert d_vector == self.d_vector
    labels = self.coarse_q.predict(input)

    unique_labels, counts = torch.unique(labels, return_counts=True)
    ioa = self.get_ioa(labels, unique_labels)

    # expand storage if necessary
    while True:
      free_space = self.div_capacity[labels] - self.div_size[labels] - (ioa + 1)
      expansion_required = labels[free_space < 0].unique()
      if expansion_required.shape[0] == 0:
        break
      self.expand(expansion_required)

    #get write address
    empty_adr = torch.nonzero(self.is_empty == 1)[:, 0] #[n_empty]
    div_of_empty_adr = self.get_div_of_address(empty_adr) #[n_empty]
    write_address = self.get_write_address(
      empty_adr=empty_adr,
      labels=labels,
      ioa=ioa
    )
    
    # quantize input and store
    code = self.encode(input) #[(n_subvectors+n_subvectors_r), n_data]
    code = code.reshape(
      (self.n_subvectors + self.n_subvectors_r) // self.n_cs,
      self.n_cs,
      n_data
    ).transpose(1, 2)
    self.set_data_of_address(code, write_address)

    #store ids
    if input_ids is None:
      if self.tot_size == 0:
        max_id = 0
      else:
        max_id = self.address2id.max()
      input_ids = torch.arange(start=max_id, end=n_data, device=input.device)
    self.address2id[write_address] = input_ids
    self.is_empty[write_address] = 0

    # update number of stored items in each division
    self.div_size[unique_labels] += counts

    if self.verbose > 1:
      print(f"{n_data} new items added")

    return input_ids

  def train(self, input, force_retrain=False):
    """
      input: torch.Tensor, shape : [d_vector, n_data], dtype : float32
      force_retrain: bool, default : False
    """
    if self._is_trained == True and not force_retrain:
      print("Module is already trained")
      return
    d_vector, n_data = input.shape
    assert d_vector == self.d_vector
    if self.distance == "cosine":
      input = self.normalize(input)

    if self.verbose > 0:
      print("Start training coarse quantizer...")
    self.coarse_q.fit(input)

    if self.verbose > 0:
      print("Start training product quantizer...")
    code = self.product_q.train_codebook(input)

    if self.use_residual:
      reconstructed = self.product_q.decode(code)
      input = input - reconstructed
      if self.verbose > 0:
        print("Start training PQ-rerank with residuals...")
    else:
      if self.verbose > 0:
        print("Start training PQ-rerank...")

    self.product_q_r.train_codebook(input)

    self._is_trained.data = torch.tensor(True)
    if self.verbose > 0:
      print("Successfully trained!")

  def encode(self, input):
    """
      input: torch.Tensor, shape : [d_vector, n_data], dtype : float32
      return: torch.Tensor, shape : [(n_subvectors + n_subvectors_r), n_data], dtype : uint8
    """
    assert self._is_trained == True, "Module is not trained"
    assert input.shape[0] == self.d_vector
    if self.distance == "cosine":
      input = self.normalize(input)
    
    code1 = self.product_q.encode(input) #[n_subvectors, n_data]
    if self.use_residual:
      reconstructed = self.product_q.decode(code1)
      input = input - reconstructed
    code2 = self.product_q_r.encode(input) #[n_subvectors_r, n_data]
    code = torch.cat([code1, code2], dim=0) #[n_subvectors + n_subvectors_r, n_data]
    return code.byte()

  def decode(self, code):
    """
      code: torch.Tensor, shape : [(n_subvectors+n_subvectors_r), n_data], dtype : uint8
      return: torch.Tensor, shape : [d_vector, n_data], dtype : float32
    """
    assert self._is_trained == True, "Module is not trained"
    assert code.shape[0] == self.n_subvectors + self.n_subvectors_r
    code1 = code[:self.n_subvectors, :] #[n_subvectors, n_data]
    code2 = code[self.n_subvectors:, :] #[n_subvectors_r, n_data]
    recon2 = self.product_q_r.decode(code2)
    if self.use_residual:
      recon1 = self.product_q.decode(code1)
      recon2 = recon2 + recon1
    return recon2
  
  @staticmethod
  def get_product(data, precomputed):
      """
        data: torch.Tensor, shape : [n_subvectors_r // n_cs, n_data, n_cs]
        precomputed: torch.Tensor, shape : [n_subvectors_r, n_query, n_pq_clusters]
      """
      o, n_data, n_cs = data.shape
      n_subvectors_r, n_query, n_pq_clusters = precomputed
      assert n_subvectors_r == o * n_cs

      arange = torch.arange(n_subvectors_r, device="cuda:0")
      data = data.transpose(1, 2).reshape(n_subvectors_r, n_data) #[n_subvectors_r, n_data]
      result = precomputed[arange, :, data[:].long() ].sum(dim=1).T #[]
      return result
  
  def topk(self, query, k, mode=2):
    """
      query: torch.Tensor, shape : [d_vector, n_query], dtype : float32
      k: int
      mode: int, default : 2
        mode 2 is generally faster, but MIGHT have slight errors
    """
    assert self._is_trained == True, "module is not trained"
    d_vector, n_query = query.shape
    assert d_vector == self.d_vector
    assert 0 < k <= self.tot_size

    # select closest cells with Coarse Q
    centroids = self.coarse_q.centroids
    sims = self.coarse_q.sim(query, centroids)
    _, topk_labels = sims.topk(k=self.n_probe, dim=1)
    # topk_labels = torch.sort(topk_labels, dim=1)[0]
    div_start = self.div_start[topk_labels].int()
    if mode == 1: div_size = self.div_capacity[topk_labels].int()
    elif mode == 2: div_size = self.div_size[topk_labels].int()

    # selection with PQ
    reshaped_query = query.reshape(self.n_subvectors, self.d_subvector, n_query)
    codebook = self.product_q.codebook #[n_subvectors, d_subvector, n_pq_clusters]
    precomputed = self.product_q.kmeans.sim(reshaped_query, codebook, normalize=False)#[n_subvectors, n_query, n_pq_clusters]

    selectedv, selected_address = self._topk_fn(
      k=k * self.rerank_factor,
      data=self.storage[:(self.n_subvectors//self.n_cs)],
      precomputed=precomputed,
      is_empty=self.is_empty,
      div_start=div_start,
      div_size=div_size,
    ) #[n_query, k * rerank_factor]
  
    # rerank with PQ-R
    selected_address = selected_address.long()
    codebook2 = self.product_q_r.codebook

    if self.use_residual:
      selected_code = self.storage[:, selected_address, :] #[(code_size)//n_cs, n_query, k*rerank+factor, n_cs]
      selected_code = (
        selected_code
        .transpose(3, 2)
        .transpose(2, 1)
        .reshape(self.code_size, n_query * k*self.rerank_factor)
      )
      selected_code1 = selected_code[:self.n_subvectors, :] #[n_subvectors, n_query*k*rerank_factor]
      selected_recon = (
        self.product_q
        .decode(selected_code1)
        .reshape(d_vector, n_query, k*self.rerank_factor)
        .transpose(0, 1)
      )

      selected_code2 = selected_code[self.n_subvectors:, :] #[n_subvectors_r, n_query*k*rerank_factor]
      selected_residual = (
        self.product_q_r
        .decode(selected_code2)
        .reshape(d_vector, n_query, k*self.rerank_factor)
        .transpose(0, 1)
      )
      selected_recon.add_(selected_residual) #[n_query, d_vector, k*rerank_factor]

      sims = self.product_q_r.kmeans.sim(
        selected_recon,
        query.T[:, :, None], 
        normalize=False
      ).squeeze(2)
      topkv, topk_selected_address = sims.topk(k=k, dim=-1)
      topk_address = torch.gather(selected_address, index=topk_selected_address.long(), dim=1)
      topki = self.get_id_of_address(topk_address)

    else:
      selected_code = self.storage[(self.n_subvectors // self.n_cs):, selected_address, :] #[n_subvectors_r//n_cs, n_query, k*rerank+factor, n_cs]
      selected_code = selected_code.reshape(
        self.n_subvectors_r // self.n_cs,
        n_query * k * self.rerank_factor,
        self.n_cs
      )
      reshaped_query2 = query.reshape(self.n_subvectors_r, self.d_subvector_r, n_query)
      precomputed2 = self.product_q_r.kmeans.sim(reshaped_query2, codebook2, normalize=False)

      is_empty2 = torch.zeros(
        n_query * k * self.rerank_factor,
        device=self.device,
        dtype=torch.uint8
      )
      div_start2 = torch.arange(
        n_query,
        device=self.device,
        dtype=torch.int32
      )[:, None] * k * self.rerank_factor
      div_size2 = torch.empty_like(div_start2)
      div_size2.fill_(k * self.rerank_factor)

      topkv, topk_selected_address = self._topk_r_fn(
        k=k,
        data=selected_code,
        precomputed=precomputed2,
        is_empty=is_empty2,
        div_start=div_start2,
        div_size=div_size2,
      ) #[n_query, k]

      topk_selected_address = topk_selected_address - div_start2
      topk_address = torch.gather(selected_address, index=topk_selected_address.long(), dim=1)
      topki = self.get_id_of_address(topk_address)

    return (topkv, topki)