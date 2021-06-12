import torch
from .BaseContainer import BaseContainer
from .. import util

class FlatContainerGroup(BaseContainer):
  def __init__(
      self,
      code_size_list,
      dtype_list,
      device_list,
      contiguous_size_list=None,
      initial_size=None,
      expand_step_size=1024,
      expand_mode="double",
      use_inverse_id_mapping=False,
      verbose=0
    ):
    assert type(code_size_list) in [list, tuple]
    n_storage = len(code_size_list)
    assert type(device_list) in [list, tuple]
    assert type(dtype_list) in [list, tuple]
    assert len(device_list) == len(dtype_list) == n_storage
    dtype_list = [util.str2dtype(i) for i in dtype_list]
    if contiguous_size_list is not None:
      assert type(contiguous_size_list) in [list, tuple]
      assert len(contiguous_size_list) == n_storage
      for i in range(n_storage):
        assert code_size_list[i] % contiguous_size_list[i] == 0
    else:
      contiguous_size_list = [1] * n_storage

    main_device = "cuda:0" if "cuda" in [torch.device(i).type for i in device_list] else "cpu"
    super(FlatContainerGroup, self).__init__(
      device=main_device,
      initial_size=initial_size,
      expand_step_size=expand_step_size,
      expand_mode=expand_mode,
      use_inverse_id_mapping=use_inverse_id_mapping
    )

    self.code_size_list = code_size_list
    self.device_list = device_list
    self.dtype_list = dtype_list
    self.contiguous_size_list = contiguous_size_list
    self.verbose = verbose
    self._n_items = 0
    n_storage = len(self.code_size_list)
    for i in range(n_storage):
      storage = torch.zeros(
        code_size_list[i] // contiguous_size_list[i],
        self.initial_size,
        contiguous_size_list[i],
        dtype = dtype_list[i],
        device = device_list[i]
      )
      self.register_buffer(f"_storage_{i}", storage)

  @property
  def n_storage(self):
    return len(self.code_size_list)

  @property
  def n_items(self):
    return self._n_items

  def __getitem__(self, index):
    module = FlatContainer(
      code_size = self.code_size_list[index],
      expand_step_size = self.expand_step_size,
      initial_size = self.initial_size,
      contiguous_size = self.contiguous_size_list[index],
      use_inverse_id_mapping = self.use_inverse_id_mapping,
      expand_mode = self.expand_mode,
      verbose = self.verbose,
      dtype = self.dtype_list[index],
      device = self.device_list[index]
    )
    module._n_items = self._n_items
    del module._storage
    del module._address2id
    del module._id2address
    
    storage = getattr(self, f"_storage_{index}")
    module.register_buffer("_storage", storage)
    module.register_buffer("_id2address", self._id2address)
    module.register_buffer("_address2id", self._address2id)

    def read_only_alert(*args, **kwargs):
      raise NotImplementedError("Module cannot be modified")
    module.add = read_only_alert
    module.remove = read_only_alert
    module.expand = read_only_alert

    return module
    
  def get_data_by_address(self, address):
    assert util.check_dtype(address, torch.int64)
    address = address.to(self.device).clone()
    
    n_address = address.shape[0]
    mask = (0 <= address) & (address < self.capacity)
    address[~mask] = 0
    result = []
    for i in range(self.n_storage):
      device = self.device_list[i]
      dtype = self.dtype_list[i]
      code_size = self.code_size_list[i]
      storage = getattr(self, f"_storage_{i}")
      data = storage.index_select(
        dim=1,
        index=address.to(device)
      )
      data.index_fill_(
        dim=1,
        index=torch.nonzero(~mask).squeeze(1).to(device),
        value=0
      )
      data = data.transpose(1, 2).reshape(code_size, -1)
      result.append(data)
    return result

  def set_data_by_address(self, address, data):
    assert util.check_dtype(address, torch.int64)
    assert type(data) in [list, tuple]
    assert len(data) == self.n_storage
    address = address.to(self.device)
    mask = (0 <= address) & (address < self.capacity)

    for i in range(self.n_storage):
      device = self.device_list[i]
      dtype = self.dtype_list[i]
      code_size = self.code_size_list[i]
      contiguous_size = self.contiguous_size_list[i]
      assert util.check_dtype(data[i], dtype)
      assert data[i].shape[0] == code_size
      data_i = data[i].to(device).reshape(
        code_size // contiguous_size,
        contiguous_size,
        -1,
      ).transpose(1, 2)
      storage = getattr(self, f"_storage_{i}")
      storage[:, address[mask].to(device)] = data_i[:, mask]

  def empty(self):
    super(FlatContainerGroup, self).empty()
    for i in range(self.n_storage):
      storage = getattr(self, f"_storage_{i}")
      storage.fill_(0)
    self._n_items = 0

  def expand(self):
    super(FlatContainerGroup, self).expand()
    n_new = self.expand_step_size
    
    # expand storage
    for i in range(self.n_storage):
      device = self.device_list[i]
      dtype = self.dtype_list[i]
      code_size = self.code_size_list[i]
      contiguous_size = self.contiguous_size_list[i]
      storage = getattr(self, f"_storage_{i}")
      delattr(self, f"_storage_{i}")
      new_storage = torch.zeros(
        code_size // contiguous_size,
        n_new,
        contiguous_size,
        device=device,
        dtype=dtype
      )
      storage = torch.cat([
        storage,
        new_storage
      ], dim=1)
      self.register_buffer(f"_storage_{i}", storage)

  def add(self, data, ids=None, return_address=False):
    assert type(data) in [list, tuple]
    assert len(data) == self.n_storage
    n_data = data[0].shape[1]

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
    # for i in range(self.n_storage):
    #   storage = getattr(self, f"_storage_{i}")
    #   device = self.device_list[i]
    #   storage[:, self.n_items:self.n_items+n_data] = data[i].to(device)
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
      util.check_dtype(address, torch.int64)
      address = address.to(self.device)
    else:
      raise RuntimeError("Need either ids or address")

    mask = (0 <= address) & (address < self.capacity)
    address = address[mask].unique(sorted=True)
    n_removed = address.shape[0]
    assert n_removed <= self.n_items
    if n_removed == 0:
      return 0

    self._address2id[address] = -1
    tail_ids = self._address2id[self.n_items-n_removed:self.n_items]
    mask = tail_ids != -1

    for i in range(self.n_storage):
      storage = getattr(self, f"_storage_{i}")
      tail_data = storage[:,self.n_items-n_removed:self.n_items] #
      n = (~mask).sum().item() # number of removed items in tail
      storage[:, address[:n_removed-n]] = tail_data[:, mask].clone()
      storage[:, self.n_items-n_removed:self.n_items] = 0

    self._address2id[address[:n_removed-n]] = tail_ids[mask].clone()
    self._address2id[self.n_items-n_removed:self.n_items] = -1
    if self.use_inverse_id_mapping:
      self.create_inverse_id_mapping()
    self._n_items -= n_removed
    return n_removed