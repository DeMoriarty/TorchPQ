import torch
import torch.nn as nn
import numpy as np

from .kernels import GetAddressOfIDCUDA
from .kernels import GetDivOfAddressCUDA
from .kernels import GetIOACUDA
from .kernels import GetWriteAddressCUDA
from .SQ import SQ
from .CustomModule import CustomModule

class IVFPQBase(CustomModule):
  def __init__(
    self,
    d_vector,
    code_size,
    n_cq_clusters,
    blocksize,
    verbose,
    distance,
    cpu_quantizer,
    device,
    ):
    """
    this is a base class for IVFPQ variants

    Parameters:
      d_vector: int
        dimentionality of vectors to be quantized.

      code_size: int, default : 8
        byte size of quantized code, must to be a multiple of 4
        maximum possible code_size depends on GPU architecture
          GPU Architecture:   fp32  fp16 
          Ampere (GA100)      163   326  
          Turing (TU102 etc.) 64    128  
          Volta (GV100)       96    192  
          Pascal and before   48    96   
      
      n_cq_clusters: int, default : 128
        number coarse quantizer clusters
        recommended value is between 4*sqrt(n_data) ~ 16*sqrt(n_data)

      blocksize: int, default : 64
        number of vectors that can be asigned to each cluster of coarse_quantizer initially
        can be expanded using .expand methods, .add method will automatically cal .expand if necessary
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
    super(IVFPQBase, self).__init__()
    self.n_cs = 4
    assert code_size % self.n_cs == 0, f"code_size needs to be a multiple of {self.n_cs}"
    
    self.d_vector = d_vector
    self.code_size = code_size
    self.extra_code_size = 9
    self.n_cq_clusters = n_cq_clusters
    self.blocksize = blocksize
    self.verbose = verbose
    self.register_buffer("_is_trained", torch.tensor(False))
    self.device = device
    self.cpu_quantizer = cpu_quantizer
    if cpu_quantizer is not None:
      if cpu_quantizer.bits in [32, 16, 8]:
        self.cpu_code_size = d_vector
      elif cpu_quantizer.bits in [4]:
        self.cpu_code_size = d_vector // 2

      if cpu_quantizer.bits == 32:
        self.cpu_dtype = torch.float32
      elif cpu_quantizer.bits == 16:
        self.cpu_dtype = torch.float16
      elif cpu_quantizer.bits in [8, 4]:
        self.cpu_dtype = torch.uint8

    if distance.lower() in ["euclidean", "l2"]:
      distance = "euclidean"
    elif distance.lower() in ["dot", "inner"]:
      distance = "inner"
    elif distance.lower() in ["cosine"]:
      distance = "cosine"
    self.distance = distance
      
    storage = torch.zeros(
      code_size // self.n_cs,
      blocksize * n_cq_clusters,
      self.n_cs,
      device=self.device,
      dtype=torch.uint8,
    )

    if cpu_quantizer is None:
      self.cpu_storage = None
    else:
      self.cpu_storage = torch.zeros(
        self.cpu_code_size,
        blocksize * n_cq_clusters,
        device="cpu",
        dtype=self.cpu_dtype,
      )

    address2id = torch.ones(
      blocksize * n_cq_clusters,
      device=self.device,
      dtype=torch.long,
    ) * -1
    is_empty = torch.ones(
      blocksize * n_cq_clusters,
      device=self.device,
      dtype=torch.uint8
    )
    div_start = torch.arange(
      n_cq_clusters,
      device=self.device
    ) * blocksize
    div_capacity = torch.ones(
      n_cq_clusters,
      device=self.device,
      dtype=torch.long,
    ) * blocksize
    div_size = torch.zeros(
      n_cq_clusters,
      device=self.device,
      dtype=torch.long
    )
    self.register_buffer("storage", storage)
    self.register_buffer("div_start", div_start)
    self.register_buffer("div_capacity",div_capacity)
    self.register_buffer("div_size", div_size)
    self.register_buffer("address2id", address2id)
    self.register_buffer("is_empty", is_empty)

    self._get_address_of_id_cuda = GetAddressOfIDCUDA(
      tpb=256,
    )
    self._get_div_of_address_cuda = GetDivOfAddressCUDA(
      ta=1,
      tpb=256,
    )
    self._get_ioa_cuda = GetIOACUDA(
      tpb=256,
    )
    self._get_write_address_cuda = GetWriteAddressCUDA(
      tpb=256,
    )

  def __getitem__(self, ids):
    """
      ids: int, or an iterable of integers
    """
    if type(ids) is not torch.Tensor:
      ids = torch.LongTensor(ids)
    if len(ids.shape) == 0:
      ids = ids[None]
    elif len(ids.shape) > 1:
      raise ValueError("only 1D array/tensors are supported")
    return self.get_data_of_id(ids)

  def __repr__(self):
    def get_bytesize_str(bytesize):
      if bytesize < 1024*1:
        bytesize = f"{bytesize} B"
      elif bytesize < 1024**2:
        bytesize = f"{round(bytesize / 1024, 2)} KB"
      elif bytesize < 1024**3:
        bytesize = f"{round(bytesize / 1024**2, 2)} MB"
      elif bytesize < 1024**4:
        bytesize = f"{round(bytesize / 1024**3, 2)} GB"
      elif bytesize < 1024**5:
        bytesize = f"{round(bytesize / 1024**4, 2)} TB"
      return bytesize

    bytesize = self.bytesize
    cpu_bytesize = get_bytesize_str(bytesize[0])
    gpu_bytesize = get_bytesize_str(bytesize[1])

    txt =  f""" {type(self).__name__}(
      number of stored items = {self.tot_size},
      total capacity of storage = {self.tot_capacity},
      byte size on CPU = {cpu_bytesize},
      byte size on GPU = {gpu_bytesize},
      code size = {self.code_size} + {self.extra_code_size} B,
      is trained = {self._is_trained.item()}
    )"""
    txt = '\n'.join([i[4:] for i in txt.split('\n')])
    return txt

  def __len__(self):
    return self.tot_size

  @staticmethod
  def get_cc():
    if torch.cuda.is_available():
      device_id = torch.cuda.current_device()
      gpu_properties = torch.cuda.get_device_properties(device_id)
      result = (gpu_properties.major, gpu_properties.minor)
    else:
      result = (10, 0)
    return result
  
  @staticmethod
  def normalize(a, dim=-1):
    """
      normalize given tensor along given dimention
    """
    a_norm = a.norm(dim=dim, keepdim=True) + 1e-9
    return a / a_norm

  @property
  def tot_size(self):
    """
      total number of items stored
    """
    return self.div_size.sum().item()

  @property
  def tot_capacity(self):
    """
      total capacity of storage
    """
    return self.div_capacity.sum().item()

  @property
  def bytesize(self):
    """
      the bytesize of cpu and gpu memory that are being used by module
      returns: (cpu_bytesize, gpu_bytesize)
    """
    cpu_bytesize = 0
    gpu_bytesize = 0
    for p in self.state_dict().values():
      if type(p) is torch.Tensor:
        if p.device.type == "cpu":
          cpu_bytesize += p.numel() * p.element_size()
        elif p.device.type == "cuda":
          gpu_bytesize += p.numel() * p.element_size()
    if self.cpu_storage is not None:
      cpu_bytesize += self.cpu_storage.numel() * self.cpu_storage.element_size()
    return (cpu_bytesize, gpu_bytesize)

  def _get_address_of_id_old(self, ids):
    batch_size = ids.shape[0]
    address = torch.zeros(batch_size, device=self.device, dtype=torch.long)
    for i in range(batch_size):
      id = ids[i]
      adr = torch.nonzero(self.address2id == id)
      if adr.shape[0] > 0:
        address[i] = adr[0, 0]
      else:
        address[i] = -1
    return address

  def get_address_of_id(self, ids):
    """
      ids: torch.Tensor, shape : [batch_size], dtype : int64
      if there are multiple matching ids, only the address of the first one will be returned
      if there are no matching ids, the address is -1
    """
    if self.address2id.device.type == "cuda":
      return self._get_address_of_id_cuda(self.address2id, ids)
    elif self.address2id.device.type == "cpu":
      return self._get_address_of_id_old(ids)

  def get_id_of_address(self, address):
    """
      address: torch.Tensor, shape : [batch_size], dtype : int64
    """
    return self.address2id[address]

  def _get_div_of_address_old(self, address):
    assert address.max() < self.tot_capacity
    batch_size = address.shape[0]
    div_start = self.div_start
    div_end = div_start + self.div_capacity
    # for adr in address:
    #   mask1 = div_start <= adr
    #   mask2 = div_ends > adr
    #   mask = mask1 & mask2
    #   div = torch.nonzero(mask)
    mask1 = div_start[None, ] <= address[:, None]
    mask2 = div_end[None, ] > address[:, None] # [batch_size, n_cq_clusters]
    mask = mask1 & mask2
    not_found = mask.sum(dim=1) == 0
    mask[not_found, 0] = True
    divs = torch.nonzero(mask)
    divs[not_found, 1] = -1
    return divs[:, 1]

  def get_div_of_address(self, address):
    """
      address: torch.Tensor, shape : [n_address], dtype : int64
      if address is not in range [0, tot_capacity), its div will be -1
    """
    if self.div_start.device.type == "cuda":
      div_start = self.div_start
      div_end = div_start + self.div_capacity
      return self._get_div_of_address_cuda(address, div_start, div_end)
    elif self.div_start.device.type == "cpu":
      return self._get_div_of_address_old(address)

  def get_data_of_address(self, address):
    """
      address: torch.Tensor, shape : [n_address], dtype : int64
      returns: torch.Tensor, shape : [code_size, n_address], dtype : uint8
      address must be in range [0, tot_capacity)
    """
    n_address = address.shape[0]
    assert address.dtype == torch.long
    mask = address < 0
    data = self.storage[:, address] #[code_size//n_cs, n_address, n_cs]
    data = data.transpose(1,2).reshape(self.code_size, n_address)
    if mask.sum() > 0:
      data[:, mask] = 0
    return data

  def get_data_of_id(self, ids):
    """
      ids: torch.Tensor, shape : [n_ids], dtype : int64
      returns: torch.Tensor, [code_size, n_ids]
      if an id doesn't exist, its retrieved data will be zeros.
    """
    address = self.get_address_of_id(ids)
    data = self.get_data_of_address(address)
    return data

  def get_cpu_data_of_address(self, address):
    """
      address: torch.Tensor, shape : [n_address], dtype : int64
      return: torch.Tensor, shape : [cpu_code_size, n_address], dtype : cpu_dtype
      address must be in range [0, tot_capacity)
    """
    if self.cpu_storage is not None:
      address = address.cpu()
      n_address = address.shape[0]
      assert address.dtype == torch.long
      mask = address < 0
      data = self.cpu_storage[:, address] #[cpu_code_size, n_address]
      if mask.sum() > 0:
        data[:, mask] = 0
      return data

  def get_cpu_data_of_id(self, ids):
    """
      ids: torch.Tensor, shape : [n_ids], dtype : int64
      returns: torch.Tensor, [cpu_code_size, n_ids]
      if an id doesn't exist, its retrieved data will be zeros.
    """
    address = self.get_address_of_id(ids)
    data = self.get_cpu_data_of_address(address)
    return data

  def set_data_of_address(self, data, address):
    """
      data: torch.Tensor, shape : [code_size // n_cs, n_data, n_cs], dtype : uint8
      address: torch.Tensor, shape : [n_data], dtype : int64
    """
    self.storage[:, address] = data

  def set_data_of_id(self, data, ids):
    """
      data: torch.Tensor, shape : [code_size // n_cs, n_data, n_cs], dtype : uint8
      ids: torch.Tensor, shape : [n_data], dtype : int64
    """
    address = self.get_address_of_id(ids)
    self.set_data_of_address(data, address)

  def set_cpu_data_of_address(self, data, address):
    """
      data: torch.Tensor, shape : [cpu_code_size, n_data, ], dtype : cpu_dtype
      address: torch.Tensor, shape : [n_data], dtype : int64
    """
    if self.cpu_storage is not None:
      assert data.dtype == self.cpu_dtype
      assert data.shape[0] == self.cpu_code_size
      address = address.cpu()
      data = data.cpu()
      self.cpu_storage[:, address] = data

  def set_cpu_data_of_id(self, data, ids):
    """
      data: torch.Tensor, shape : [code_size // n_cs, n_data, n_cs], dtype : uint8
      ids: torch.Tensor, shape : [n_data], dtype : int64
    """
    if self.cpu_storage is not None:
      address = self.get_address_of_id(ids)
      self.set_cpu_data_of_address(data, address)

  def save_cpu_data(self, path):
    if self.cpu_quantizer is not None:
      torch.save(self.cpu_storage, path)

  def load_cpu_data(self, path):
    if self.cpu_quantizer is not None:
      self.cpu_storage = torch.load(path)

  def load_data(self, path):
    state_dict = torch.load(path)
    self.load_state_dict(state_dict)

  def _get_ioa_old(self, labels, unique_labels=None):
    if unique_labels is None:
      unique_labels = torch.unique(labels) #[n_unique_clusters]
    expanded_labels = labels[:, None].expand(-1, unique_labels.shape[0]) #[n_data, n_unique_clusters]
    mask = expanded_labels == unique_labels[None, :] #[n_data, n_unique_clusters]
    mcs = mask.cumsum(dim=0)
    mcs[[~mask]] = 0
    ioa = mcs.sum(dim=1) - 1

  def get_ioa(self, labels, unique_labels=None):
    if labels.device.type == "cuda":
      return self._get_ioa_cuda(labels, unique_labels)
    elif labels.device.type == "cpu":
      return self._get_ioa_old(labels, unique_labels)

  def _get_write_address_old(self, empty_adr, div_of_empty_adr, labels, ioa):
    write_address = torch.zeros_like(labels)
    for i in range(n_labels):
      div_mask = div_of_empty_adr == labels[i]
      write_adr = empty_adr[div_mask]
      write_adr = write_adr[ioa[i] ]
      write_address[i] = write_adr
    return write_address

  def get_write_address(self, empty_adr, labels, div_of_empty_adr=None, ioa=None):
    if div_of_empty_adr is None:
      div_of_empty_adr = self.get_div_of_address(empty_adr)
    if ioa is None:
      ioa = self.get_ioa(labels)

    if labels.device.type == "cuda":
      return self._get_write_address_cuda(empty_adr, div_of_empty_adr, labels, ioa)
    elif labels.device.type == "cpu":
      return self._get_write_address_old(empty_adr, div_of_empty_adr, labels, ioa)

  def expand(self, clusters):
    """
        clusters: an iterable of integers, len(clusters) <= n_cq_clusters
    """
    if self.verbose > 1:
      print(f"Expanding {len(clusters)} clusters")
    tot = 0
    storage = self.storage
    address2id = self.address2id
    is_empty = self.is_empty
    cpu_storage = self.cpu_storage
    del self.storage, self.address2id, self.is_empty, self.cpu_storage
    for cluster_index in clusters:
      div_start = self.div_start[cluster_index]
      div_cap = self.div_capacity[cluster_index].item()
      div_end = div_start + div_cap

      new_block = torch.zeros(
        self.code_size // self.n_cs,
        div_cap,
        self.n_cs, device=self.device,
        dtype=torch.uint8,
      )
      storage = torch.cat([
        storage[:, :div_end],
        new_block,
        storage[:, div_end:]
      ], dim=1)
      
      if cpu_storage is not None:
        cpu_new_block = torch.zeros(
          self.cpu_code_size,
          div_cap,
          device="cpu",
          dtype=self.cpu_dtype
        )
        cpu_storage = torch.cat([
          cpu_storage[:, :div_end],
          cpu_new_block,
          cpu_storage[:, div_end:]
        ], dim=1)

      new_a2i = torch.ones(div_cap, device=self.device, dtype=address2id.dtype) * -1
      address2id = torch.cat([
        address2id[:div_end],
        new_a2i,
        address2id[div_end:]
      ], dim=0)

      new_is_empty = torch.ones(div_cap, device=self.device, dtype=is_empty.dtype)
      is_empty = torch.cat([
        is_empty[:div_end],
        new_is_empty,
        is_empty[div_end:]
      ], dim=0)

      self.div_capacity[cluster_index] += div_cap
      arange = torch.arange(start=cluster_index+1, end=self.n_cq_clusters)
      self.div_start[arange] += div_cap
      # self.div_start[:] = self.div_capacity.cumsum(dim=0)
      tot += div_cap
    self.register_buffer("storage", storage)
    self.register_buffer("address2id", address2id)
    self.register_buffer("is_empty", is_empty)
    self.cpu_storage = cpu_storage
    if self.verbose > 1:
      print(f"Storage capacity is increased by {tot}, total capacity: {self.storage.shape[1]}")
  
  def add_to_cpu_ram(self, input, address):
    """
      add scalar quantized input to cpu ram.
    """
    if self.cpu_quantizer is not None:
      code = self.cpu_quantizer.encode(input)
      self.set_cpu_data_of_address(code, address)

  def add(self, input, input_ids=None):
    """
      has to be overrided in child class
    """
    raise NotImplementedError("method needs to be overrided in child class")

  def remove_address(self, address):
    assert self._is_trained == True, "Module is not trained"
    n_address = address.shape[0]
    address = address[address >= 0] #filter out -1 values
    divs = self.get_div_of_address(address)

    self.is_empty[address] = 1
    self.address2id[address] = -1

    unique_divs, counts = divs.unique(return_counts=True)
    self.div_size[unique_divs] -= counts
    if self.verbose > 1:
      print(f"{len(address)} items have been successfully removed, {n_address - len(address)} items are ignored.")

  def remove(self, remove_ids):
    """
      remove_ids: torch.Tensor, shape : [n_ids], dtype : int64
    """
    assert self._is_trained == True, "Module is not trained"
    address = self.get_address_of_id(remove_ids)
    self.remove_address(address)
    
  def train(self, input, force_retrain=False):
    """
      has to be overrided in child class
    """
    raise NotImplementedError("method needs to be overrided in child class")

  def encode(self, input):
    """
      has to be overrided in child class
    """
    raise NotImplementedError("method needs to be overrided in child class")

  def decode(self, code):
    """
      has to be overrided in child class
    """
    raise NotImplementedError("method needs to be overrided in child class")
    
  def topk(self, query, k, mode=1):
    """
      has to be overrided in child class
    """
    raise NotImplementedError("method needs to be overrided in child class")
    