import torch
import numpy as np

from .IVFPQBase import IVFPQBase
from .IVFPQTopk import IVFPQTopk
from .PQ import PQ
from .kmeans import KMeans

class IVFPQ(IVFPQBase):
  def __init__(
    self,
    d_vector,
    n_subvectors=8,
    n_cq_clusters=128,
    n_pq_clusters=256,
    blocksize=64,
    verbose=0,
    distance="euclidean",
    cpu_quantizer=None,
    device='cuda:0'
    ):
    """
    this is an efficient implementation of IVFPQ algorithm
    for more details, you can search 'Product Quantization for Nearest Neighbor Search'

    Parameters:
      d_vector: int
        dimentionality of vectors to be quantized.

      n_subvectors: int, default : 8
        number of sub-quantizers, needs to be a multiple of 4,
        maximum possible n_subvectors depends on GPU architecture
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

      cpu_quantizer: SQ or None, default : None
        scalar quantizer used to quantize datapoints to be stored in CPU RAM
        if None is given, CPU RAM will not be used.

      device: str, default : 'cuda:0'
        which device to use
        can be one of: ['cpu', 'cuda', 'cuda:X'] *X represents cuda device index
      
    """
    n_cs = 4
    self.n_cs = n_cs
    assert n_subvectors % self.n_cs == 0, f"n_subvectors needs to be a multiple of {n_cs}"
    self.n_probe = 1
    self.d_vector = d_vector
    self.d_subvector = d_vector // n_subvectors
    self.n_pq_clusters = n_pq_clusters
    self.n_subvectors = n_subvectors
    
    cc = self.get_cc() #compute capability
    if cc[0] < 7 or cc == (7, 2):
      assert n_subvectors <= 48
    elif cc == (7, 0):
      assert n_subvectors <= 96
    elif cc == (7, 5):
      assert n_subvectors <= 64
    elif cc == (8, 0):
      assert n_subvectors <= 163
    elif cc == (8, 6):
      assert n_subvectors <= 99
    
    super(IVFPQ, self).__init__(
      d_vector=d_vector,
      code_size=n_subvectors,
      n_cq_clusters=n_cq_clusters,
      blocksize=blocksize,
      verbose=verbose,
      distance=distance,
      cpu_quantizer=cpu_quantizer,
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

    self._topk_fn = IVFPQTopk(
      n_subvectors=n_subvectors,
      n_clusters=n_pq_clusters,
      n_cs=n_cs
    )

  def set_coarse_q_max_iter(self, value):
    self.coarse_q.max_iter = value

  def set_product_q_max_iter(self, value):
    self.product_q.kmeans.max_iter = value
  
  def set_coarse_q_n_redo(self, value):
    self.coarse_q.n_redo = value

  def set_product_q_n_redo(self, value):
    self.product_q.kmeans.n_redo = value

  def add(self, input, input_ids=None, return_address=False):
    """
      input: torch.Tensor, shape : [n_data, d_vector], dtype : float32
      input_ids: torch.Tensor, shape : [n_data], dtype : int64
      return_address: bool, default : False
        if set to True, return the address of added items
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
    quantized_input = self.product_q.encode(input).byte() #[n_subvector, n_data]
    quantized_input = quantized_input.reshape(
      self.n_subvectors // self.n_cs,
      self.n_cs,
      n_data
    ).transpose(1, 2)
    self.set_data_of_address(quantized_input, write_address)
    if self.cpu_quantizer is not None:
      self.add_to_cpu_ram(input, write_address)

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

    if return_address:
      return (input_ids, write_address)
    else:
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

    if self.cpu_quantizer is not None:
      if self.verbose > 0:
        print("Start training scalar quantizer")
      self.cpu_quantizer.train(input)

    if self.verbose > 0:
      print("Start training coarse quantizer...")
    self.coarse_q.fit(input)

    if self.verbose > 0:
      print("Start training product quantizer...")
    self.product_q.train(input)

    self._is_trained.data = torch.tensor(True)
    if self.verbose > 0:
      print("Successfully trained!")
  
  def encode(self, input):
    """
      input: torch.Tensor, shape : [d_vector, n_data], dtype : float32
      return: torch.Tensor, shape : [n_subvectors, n_data], dtype : uint8
    """
    assert self._is_trained == True, "Module is not trained"
    assert input.shape[0] == self.d_vector
    if self.distance == "cosine":
      input = self.normalize(input)
    code = self.product_q.encode(input)
    return code

  def decode(self, code):
    """
      code: torch.Tensor, shape : [n_subvectors, n_data], dtype : uint8
      return: torch.Tensor, shape : [d_vector, n_data], dtype : float32
    """
    assert self._is_trained == True, "Module is not trained"
    assert code.shape[0] == self.n_subvectors
    recon = self.product_q.decode(code)
    return recon
    
  def similarity_at_address(self, query, address):
    """
      computes similarity of each query and datapoint at each address
      query: torch.Tensor, shape : [d_vector, n_query], dtype : float32
      address: torch.Tensor, shape : [n_query, n_address], dtype : int64
    """
    assert self._is_trained == True, "module is not trained"
    d_vector, n_query = query.shape
    n_address = address.shape[1]
    assert d_vector == self.d_vector
    assert n_query == address.shape[0]

    data = self.storage[:, address, :] #[n_subvectors//n_cs, n_query, n_address, n_cs]
    data = data.reshape(
      self.code_size // self.n_cs,
      n_query * n_address,
      self.n_cs
    )
    precomputed = self.product_q.precompute_adc(query)

    is_empty = torch.zeros(
      n_query * n_address,
      device=self.device,
      dtype=torch.uint8
    )
    div_start = torch.arange(
      n_query,
      device=self.device,
      dtype=torch.int32
    )[:, None] * n_address
    div_size = torch.empty_like(div_start)
    div_size.fill_(n_address)

    values, indices = self._topk_fn.get_similarity(
      data=data,
      precomputed=precomputed,
      is_empty=is_empty,
      div_start=div_start,
      div_size=div_size,
    ) #[n_query, k]

    indices = indices - div_start
    values = torch.gather(values, index=indices.long(), dim=1)
    return values

    # address = torch.gather(address, index=indices.long(), dim=1)
    # ids = self.get_id_of_address(address)
    # values, ids

  def similarity_at_id(self, query, ids):
    """
      computes similarity of queries and datapoints speicified by id
      query: torch.Tensor, shape : [d_vector, n_query], dtype : float32
      ids: torch.Tensor, shape : [n_query, n_id], dtype : int64
    """
    n_query, n_ids = ids.shape

    ids = ids.reshape(n_query * n_ids)
    address = self.get_address_of_id(ids)
    address = address.reshape(n_query, n_ids)

    return self.similarity_at_address(query, address)

  def topk(self, query, k, mode=2, return_address=False):
    """
      query: torch.Tensor, shape : [d_vector, n_query], dtype : float32
      k: int
      mode: int, default : 2
        mode 2 is generally faster, but MIGHT have slight errors
      return_address: bool, default : False
    """
    assert self._is_trained == True, "module is not trained"
    d_vector, n_query = query.shape
    assert d_vector == self.d_vector
    assert 0 < k <= self.tot_size

    centroids = self.coarse_q.centroids
    sims = self.coarse_q.euc_sim(query, centroids)
    _, topk_labels = sims.topk(k=self.n_probe, dim=1)
    # topk_labels = torch.sort(topk_labels, dim=1)[0]
    div_start = self.div_start[topk_labels].int()
    if mode == 1: div_size = self.div_capacity[topk_labels].int()
    elif mode == 2: div_size = self.div_size[topk_labels].int()

    precomputed = self.product_q.precompute_adc(query)

    topkv, topk_address = self._topk_fn(
      k=k,
      data=self.storage,
      precomputed=precomputed,
      is_empty=self.is_empty,
      div_start=div_start,
      div_size=div_size,
    )
    topk_ids = self.get_id_of_address(topk_address.long())
    if return_address:
      return (topkv, topk_ids, topk_address)
    else:
      return (topkv, topk_ids)
      