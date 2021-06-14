#@title BaseContainer
import torch
from abc import ABC
from .. import util
from ..kernels import GetAddressByIDCuda
from ..CustomModule import CustomModule

class BaseContainer(CustomModule, ABC):
  def __init__(
      self,
      device="cpu",
      initial_size=None,
      expand_step_size=1024,
      expand_mode="double",
      use_inverse_id_mapping=False
    ):
    super(BaseContainer, self).__init__()
    if initial_size is None:
      initial_size = expand_step_size
    assert expand_mode in ["step", "double"]
    assert initial_size >= 0
    assert expand_step_size > 0

    self.device = device
    self.initial_size = initial_size
    self.expand_step_size = expand_step_size
    self.expand_mode = expand_mode
    self.use_inverse_id_mapping = use_inverse_id_mapping

    _address2id = torch.ones(
      initial_size,
      device=device,
      dtype=torch.long
    ) * -1
    self.register_buffer("_address2id", _address2id)
    self.register_buffer("_id2address", None)

    self._get_address_by_id_cuda = GetAddressByIDCuda(
      tpb=256
    )

  @property
  def capacity(self):
    return self._address2id.shape[0]

  @property
  def max_id(self):
    return self._address2id.max().item() 

  def empty(self):
    self._address2id.fill_(-1)
    del self._id2address
    self.register_buffer("_id2address", None)

  def get_id_by_address(self, address):
    assert util.check_dtype(address, torch.int64)
    address = address.to(self.device)

    mask = (0 <= address) & (address < self.capacity)
    ids = torch.ones_like(address) * -1
    ids[mask] = self._address2id[address[mask]]
    return ids

  def _get_address_by_id_cpu(self, ids):
    n_ids = ids.shape[0]
    address = torch.zeros(n_ids, device=self.device, dtype=torch.long)
    for i in range(n_ids):
      id = ids[i]
      adr = torch.nonzero(self._address2id == id)
      if adr.shape[0] > 0:
        address[i] = adr[0, 0]
      else:
        address[i] = -1
    return address

  def get_address_by_id(self, ids):
    assert util.check_dtype(ids, torch.int64)
    ids = ids.to(self.device)

    if self.use_inverse_id_mapping:
      if self._id2address is None:
        self.create_inverse_id_mapping()

      # assume ids are non-negative
      mask = (0 <= ids) & (ids <= self.max_id)
      address = torch.ones_like(ids) * -1
      address[mask] = self._id2address[ids[mask]]
    else:
      if ids.device.type == "cpu":
        address = self._get_address_by_id_cpu(ids)
      elif ids.device.type == "cuda":
        address = self._get_address_by_id_cuda(self._address2id, ids)
      else:
        raise RuntimeError(f"Unrecognized device {ids.device.type}")
    return address

  def create_inverse_id_mapping(self):
    del self._id2address
    a2i_v, a2i_i = self._address2id.sort()
    a2i_mask = a2i_v >= 0
    _id2address = torch.ones(
      self.max_id + 1,
      device=self.device,
      dtype=torch.long
    ) * -1
    _id2address[a2i_v[a2i_mask]] = a2i_i[a2i_mask]
    self.register_buffer("_id2address", _id2address)

  def expand(self):
    if self.expand_mode == "double":
      self.expand_step_size *= 2
    
    _address2id = self._address2id
    del self._address2id
    new_a2i = torch.ones(
      self.expand_step_size,
      device=self.device,
      dtype=torch.long
    ) * -1
    _address2id = torch.cat([
      _address2id,
      new_a2i
    ], dim=0)
    self.register_buffer("_address2id", _address2id)

  @abstractmethod
  def add(self):
    pass

  @abstractmethod
  def remove(self):
    pass