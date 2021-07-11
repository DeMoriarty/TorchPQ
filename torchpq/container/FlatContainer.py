import torch
from .BaseContainer import BaseContainer
from .. import util

class FlatContainer(BaseContainer):
  def __init__(
      self,
      code_size,
      contiguous_size = 1,
      dtype = "float32",
      device = "cpu",
      initial_size = None,
      expand_step_size = 1024,
      expand_mode = "double",
      use_inverse_id_mapping = False,
      verbose = 0,
    ):
    super(FlatContainer, self).__init__(
      device=device,
      initial_size=initial_size,
      expand_step_size=expand_step_size,
      expand_mode=expand_mode,
      use_inverse_id_mapping=use_inverse_id_mapping
    )

    if type(dtype) == str:
      dtype = util.str2dtype(dtype)
    assert code_size > 0
    assert code_size % contiguous_size == 0

    self.code_size = code_size
    self.contiguous_size = contiguous_size
    self.dtype = dtype
    self.verbose = verbose
    self._n_items = 0

    _storage = torch.zeros(
      code_size // contiguous_size,
      self.initial_size,
      contiguous_size,
      dtype=dtype,
      device=device
    )
    self.register_buffer("_storage", _storage)

  @property
  def n_items(self):
    return self._n_items

  def get_data_by_address(self, address):
    assert util.check_dtype(address, torch.int64)
    address = address.to(self.device).clone()

    n_address = address.shape[0]
    mask = (0 <= address) & (address < self.capacity)
    address[~mask] = 0
    data = self._storage.index_select(
      dim=1,
      index=address
    )
    data.index_fill_(
      dim=1,
      index=torch.nonzero(~mask).squeeze(1),
      value=0
    )
    # data = torch.zeros(
    #   self.code_size,
    #   n_address,
    #   device=self.device,
    #   dtype=self.dtype
    # )
    # data[:, mask] = self._storage[:, address[mask]]
    data = data.transpose(1, 2).reshape(self.code_size, -1)
    return data

  def set_data_by_address(self, address, data):
    assert util.check_dtype(address, torch.int64)
    assert util.check_dtype(data, self.dtype)
    assert data.shape[0] == self.code_size
    assert data.shape[1] == address.shape[0]
    address = address.to(self.device)
    data = data.to(self.device)
    data = data.reshape(
      self.code_size // self.contiguous_size,
      self.contiguous_size
      -1,
    ).transpose(1, 2)

    mask = (0 <= address) & (address < self.capacity)
    self._storage[:, address[mask]] = data[:, mask]

  def empty(self):
    super(FlatContainer, self).empty()
    self._storage.fill_(0)
    self._n_items = 0

  def expand(self):
    super(FlatContainer, self).expand()
    n_new = self.expand_step_size
    
    # expand storage
    _storage = self._storage
    del self._storage
    new_storage = torch.zeros(
      self.code_size // self.contiguous_size,
      n_new,
      self.contiguous_size,
      device=self.device,
      dtype=self.dtype
    )
    _storage = torch.cat([
      _storage,
      new_storage
    ], dim=1)
    self.register_buffer("_storage", _storage)

  def add(self, data, ids=None, return_address=False):
    assert util.check_dtype(data, self.dtype)
    assert data.shape[0] == self.code_size
    data = data.to(self.device)
    n_data = data.shape[1]

    if ids is not None:
      assert util.check_dtype(ids, torch.int64)
      assert ids.shape[0] == n_data
      ids = ids.to(self.device)
    else:
      ids = torch.arange(
        n_data,
        device=self.device,
        dtype=torch.int64
      ) + self.max_id + 1
    
    # expand if necessary
    while (self.capacity - self.n_items) < n_data:
      self.expand()

    # add data to storage
    write_address = torch.arange(
      n_data,
      device=self.device,
      dtype=torch.int64
    ) + self.n_items
    self.set_data_by_address(data=data, address=write_address)
    self._address2id[write_address] = ids

    ### OR
    # self._storage[:, self.n_items:self.n_items+n_data] = data
    # self._address2id[self.n_items:self.n_items+n_data] = ids
    
    if self.use_inverse_id_mapping:
      self.create_inverse_id_mapping()
    self._n_items += n_data
    
    if return_address:
      return (ids, write_address)
    else:
      return ids

  def remove(self, ids=None, address=None):
    if ids is not None:
      address = self.get_address_by_id(ids)
    elif address is not None:
      address = address.to(self.device)
      # util.check_device(address, self.device)
      util.check_dtype(address, torch.int64)
    else:
      raise RuntimeError("Need either ids or address")

    mask = (0 <= address) & (address < self.capacity)
    address = address[mask].unique(sorted=True)
    n_removed = address.shape[0]
    assert n_removed <= self.n_items
    if n_removed == 0:
      return 0

    self._address2id[address] = -1
    tail_data = self._storage[:,self.n_items-n_removed:self.n_items, :]
    tail_ids = self._address2id[self.n_items-n_removed:self.n_items]
    mask = tail_ids != -1
    n = (~mask).sum().item() # number of removed items in tail
    self._storage[:, address[:n_removed-n]] = tail_data[:, mask].clone()
    self._storage[:, self.n_items-n_removed:self.n_items] = 0

    self._address2id[address[:n_removed-n]] = tail_ids[mask].clone()
    self._address2id[self.n_items-n_removed:self.n_items] = -1
    if self.use_inverse_id_mapping:
      self.create_inverse_id_mapping()
    self._n_items -= n_removed
    return n_removed
