#@title CellContainer
from typing import List
import torch
from .. import util
from ..kernels import GetDivByAddressV2Cuda
from ..kernels import GetIOACuda
from ..kernels import GetWriteAddressV2Cuda
# from .BaseContainer import BaseContainer
from ..CustomModule import CustomModule

class BufferList(CustomModule):
  def __init__(self, buffer_list):
    super().__init__()
    assert type(buffer_list) in [list, tuple]
    self.n_buffers = len(buffer_list)
    for i in range(self.n_buffers):
      self.register_buffer(f"{i}", buffer_list[i])

  def __getitem__(self, key):
    assert hasattr(self, str(key)), f"item at index {key} does not exist"
    return getattr(self, str(key))

  def __setitem__(self, key, value):
    if hasattr(self, str(key)):
      buffer = getattr(self, str(key))
      buffer[:] = value
    else:
      self.register_buffer(f"{key}", value)

  def __delitem__(self, key):
    delattr(self, str(key))

  def __iter__(self):
    for i in range(self.n_buffers):
      yield getattr(self, str(i))

  def get_ptrs(self):
    return [buffer.data_ptr() for buffer in self]

class DistributedCellContainer(CustomModule):
  def __init__(
      self,
      code_size,
      n_cells,
      contiguous_size=1,
      dtype="float32",
      device="cpu",
      initial_size=None,
      expand_step_size=1024,
      expand_mode="double",
      use_inverse_id_mapping=False,
      verbose=0
    ):
    super().__init__()
    assert code_size >= 1
    assert n_cells >= 1
    assert contiguous_size >= 1
    assert code_size % contiguous_size == 0
    if type(dtype) is str:
      dtype = util.str2dtype(dtype)
    device_type = torch.device(device).type
    assert device_type in ["cpu", "cuda"]
    assert expand_mode in ["step", "double", "tight"]
    if initial_size is None:
      initial_size = expand_step_size
    assert initial_size >= 0
    assert expand_step_size >= 1

    self.code_size = code_size
    self.n_cells = n_cells
    self.contiguous_size = contiguous_size
    self.dtype = dtype
    self.device = device
    self.device_type = device_type
    self.initial_size = initial_size
    self.expand_step_size = expand_step_size
    self.expand_mode = expand_mode
    self.use_inverse_id_mapping = use_inverse_id_mapping
    self.verbose = verbose
    self._capacity = initial_size * n_cells
    self._n_items = 0
    self._max_id = -1

    storage = [
      torch.empty(
        code_size // contiguous_size,
        initial_size,
        contiguous_size,
        device=device,
        dtype=dtype
      )
      for _ in range(n_cells)
    ]
    self._storage = BufferList(storage)

    address2id = [
      torch.zeros(
        initial_size,
        device=device,
        dtype=torch.int64
      ) - 1
      for _ in range(n_cells)
    ]
    self._address2id = BufferList(address2id)

    cell_ptr = self._storage.get_ptrs()
    cell_ptr = torch.tensor(
      cell_ptr,
      dtype=torch.int64,
      device=device
    )
    self.register_buffer("_cell_ptr", cell_ptr)

    cell_size = torch.zeros(
      n_cells,
      device = device,
      dtype = torch.int64
    )
    self.register_buffer("_cell_size", cell_size)

    self.register_buffer("_id2address", None)

  @property
  def capacity(self):
    return self._capacity

  @property
  def n_items(self):
    return self._n_items

  @property
  def max_id(self):
    return self._max_id

  @property
  def _cell_capacity(self):
    return [i.shape[0] for i in self._address2id]

  def _reshape_data(self, data):
    """
      data:
        shape: [code_size, n_data]
        or 
        shape: [code_size//contiguous_size, n_data, contiguous_size]
    """
    if len(data.shape) == 2:
      assert data.shape[0] == self.code_size
      return data.reshape(
        self.code_size // self.contiguous_size,
        self.contiguous_size,
        -1,
      ).transpose(-1, -2)

    elif len(data.shape) == 3:
      assert data.shape[0] == self.code_size // self.contiguous_size
      assert data.shape[2] == self.contiguous_size
      return data.transpose(-1, -2).reshape(
        self.code_size,
        -1
      )
    else:
      raise RuntimeError("invalid tensor shape")

  def _get_id_by_address_cpu(self, address):
    assert util.check_dtype(address, "int64")
    assert len(address.shape) == 2
    assert address.shape[1] == 2

    n_address = address.shape[0]
    cells = address[:, 0]
    item_idx = address[:, 1]
    unique_cells = cells.unique()
    ids = torch.zeros(
      n_address,
      device=address.device,
      dtype=torch.int64
    ) - 1
    for cell in unique_cells:
      cell = cell.item()
      cell_size = self._cell_size[cell]
      adr2id_cell = self._address2id[cell]
      cell_capacity = adr2id_cell.shape[0]
      
      mask = cells == cell
      selected_item_idx = item_idx[mask].clone()
      item_mask = (0 <= selected_item_idx) & (selected_item_idx < cell_size)
      ids_mask = ids[mask].clone()
      ids_mask[item_mask] = adr2id_cell[selected_item_idx][item_mask]
      ids[mask] = ids_mask
    return ids

  def get_id_by_address(self, address):
    """
      parameters:
        address: torch.Tensor
          dtype: int64
          shape: [n_address, 2]

      return:
        ids: torch.Tensor
          dtype: int64
          shape: [n_address]
    """
    if address.device.type == "cpu":
      return self._get_id_by_address_cpu(address)
    elif address.device.type == "cuda":
      return self._get_id_by_address_cpu(address)
      # return self._get_id_by_address_cuda(address)

  def _get_address_by_id_mapped(self, ids):
    assert util.check_dtype(ids, "int64")
    assert len(ids.shape) == 1
    n_ids = ids.shape[0]
    mask = (0 <= ids) & (ids <= self.max_id)
    address = torch.zeros(
      n_ids,
      2,
      dtype = torch.int64,
      device = ids.device
    ) * -1
    address[mask] = self._id2address[ids[mask]]
    return address

  def _get_address_by_id_cpu(self, ids):
    assert util.check_dtype(ids, "int64")
    assert len(ids.shape) == 1
    n_ids = ids.shape[0]
    address = torch.zeros(
      n_ids,
      2,
      dtype = torch.int64,
      device = ids.device
    )
    for i in range(n_ids):
      id = ids[i]
      for j in range(self.n_cells):
        adr2id_cell = self._address2id[j]
        adr = torch.nonzero(adr2id_cell == id)[:, 0]
        if adr.shape[0] > 0:
          address[i, 0] = j
          address[i, 1] = adr[0]
    return address

  def get_address_by_id(self, ids):
    """
      parameters:
        ids: torch.Tensor
          dtype: int64
          shape: [n_ids]

      return:
        address: torch.Tensor
          dtype: int32
          shape: [n_ids, 2]
    """
    if self.use_inverse_id_mapping:
      if self._id2address is None:
        self.create_inverse_id_mapping()
      return self._get_address_by_id_mapped(ids)

    if ids.device.type == "cpu":
      return self._get_address_by_id_cpu(ids)
    elif ids.device.type == "cuda":
      return self._get_address_by_id_cpu(ids)

  def _get_data_by_address_cpu(self, address):
    assert util.check_dtype(address, "int64")
    assert len(address.shape) == 2
    assert address.shape[1] == 2

    n_address = address.shape[0]
    cells = address[:, 0]
    item_idx = address[:, 1]
    unique_cells = cells.unique()

    data = torch.empty(
      self.code_size // self.contiguous_size,
      n_address,
      self.contiguous_size,
      device = self.device,
      dtype = self.dtype
    )

    for cell in unique_cells:
      cell = cell.item()
      cell_size = self._cell_size[cell]
      storage_cell = self._storage[cell]
      cell_capacity = storage_cell.shape[1]
      
      mask = cells == cell
      selected_item_idx = item_idx[mask].clone()
      item_mask = (0 <= selected_item_idx) & (selected_item_idx < cell_size)

      masked_data = data[:, mask].clone()
      # print("cell_size", cell_size)
      # print("storage_cell", storage_cell.shape)
      # print("selected_item_idx", selected_item_idx)
      # print("item_mask", item_mask)
      # print("masked_data", masked_data.shape)
      masked_data[:, item_mask] = storage_cell[:, selected_item_idx][:, item_mask]
      data[:, mask] = masked_data
    data = self._reshape_data(data)
    return data

  def get_data_by_address(self, address):
    """
      parameters:
        address: torch.Tensor
          dtype: int64
          shape: [n_address, 2]

      return:
        data: torch.Tensor
          dtype: dtype
          shape: [d_vector, n_address]
    """
    if address.device.type == "cpu":
      return self._get_data_by_address_cpu(address)
    elif address.device.type == "cuda":
      return self._get_data_by_address_cpu(address)

  def _set_data_by_address_cpu(self, data, address):
    assert util.check_dtype(address, "int64")
    assert util.check_dtype(data, self.dtype)
    assert len(address.shape) == 2
    assert len(data.shape) == 2
    assert address.shape[1] == 2
    assert data.shape[0] == self.code_size
    assert data.shape[1] == address.shape[0]

    n_address = address.shape[0]
    cells = address[:, 0]
    item_idx = address[:, 1]
    unique_cells = cells.unique()
    data = self._reshape_data(data)

    for cell in unique_cells:
      cell = cell.item()
      cell_size = self._cell_size[cell]
      storage_cell = self._storage[cell]

      mask = cells == cell
      selected_item_idx = item_idx[mask].clone()
      item_mask = (0 <= selected_item_idx) & (selected_item_idx < cell_size)

      temp = storage_cell[:, selected_item_idx]
      temp[item_mask] = data[:, mask][item_mask]
      storage_cell[:, selected_item_idx] = temp
  
  def set_data_by_address(self, data, address):
    """
      parameters:
        data: torch.Tensor
          dtype: dtype
          shape: [d_vector, n_address]

        address: torch.Tensor
          dtype: int32
          shape: [n_address, 2]
    """
    if address.device.type == "cpu":
      return self._set_data_by_address_cpu(data, address)
    elif address.device.type == "cuda":
      return self._set_data_by_address_cpu(data, address)
  
  def create_inverse_id_mapping(self):
    del self._id2address
    id2address = torch.ones(
      self.max_id + 1,
      2,
      device=self.device,
      dtype=torch.int64
    ) * -1
    for i in range(self.n_cells):
      cell_size = self._cell_size[i]
      adr2id_cell = self._address2id[i][:cell_size].clone()
      arange = torch.arange(cell_size, device=self.device)
      mask = adr2id_cell >= 0
      id2address[adr2id_cell[mask], 0] = i
      id2address[adr2id_cell[mask], 1] = arange[mask]
    self.register_buffer("_id2address", id2address)

  def empty(self):
    for i in range(self.n_cells):
      storage_cell = self._storage[i]
      adr2id_cell = self._address2id[i]
      storage_cell.fill_(0)
      adr2id_cell.fill_(-1)
    self._cell_size.fill_(0)
    self._n_items = 0
    self._max_id = -1

  def expand_cell(self, cell_index):
    # cell_size = self._cell_size[cell_index]
    cell_cap = self._storage[cell_index].shape[1]
    cell_content = self._storage[cell_index]
    cell_adr2id = self._address2id[cell_index]
    del self._storage[cell_index]
    del self._address2id[cell_index]

    if self.expand_mode == "step":
      n_new = self.expand_step_size
    elif self.expand_mode == "double":
      n_new = cell_cap
    
    new_storage = torch.zeros(
      self.code_size // self.contiguous_size,
      cell_cap + n_new,
      self.contiguous_size,
      device = self.device,
      dtype = self.dtype
    )
    new_storage[:, :cell_cap] = cell_content
    self._storage[cell_index] = new_storage
    self._cell_ptr[cell_index] = new_storage.data_ptr()

    new_adr2id = torch.zeros(
      cell_cap + n_new,
      device = self.device,
      dtype = torch.int64
    ) - 1
    new_adr2id[:cell_cap] = cell_adr2id
    self._address2id[cell_index] = new_adr2id
    return n_new

  def expand(self, cells):
    n_cells = len(cells)
    total = 0
    for cell_index in cells:
      n_new = self.expand_cell(cell_index)
      total += n_new
    self.print_message(f"Total storage capacity is expanded by {total} for {n_cells} cells", 2)

  def add(self, data, cells, ids=None, return_address=False):
    """
      parameters:
        data: torch.Tensor
          shape : [code_size, n_data]
          dtype : dtype
          data to be stored

        cells: torch.Tensor
          shape : [n_data]
          dtype : int64
          cell indices of each datapoint

        ids (optional): torch.Tensor
          shape : [n_data]
          dtype : int64
          ids of each datapoint

        return_address: bool
          default : False

      return:
        ids: torch.Tensor
          shape : [n_data]
          dtype : int64
        
        address (optional): torch.Tensor
          shape : [n_data, 2]
          dtype : int64
    
    """
    assert util.check_dtype(data, self.dtype)
    assert util.check_dtype(cells, torch.int64)
    assert data.shape == (self.code_size, cells.shape[0])
    data = data.to(self.device)
    cells = cells.to(self.device)
    n_data = data.shape[1]

    if ids is not None:
      assert util.check_dtype(ids, torch.int64)
      assert ids.shape[0] == n_data
      ids = ids.to(self.device)

    else:
      ids = torch.arange(
        n_data,
        device=self.device,
        dtype=torch.int64,
      ) + self.max_id + 1

    unique_cells, unique_cell_counts = cells.unique(return_counts=True)
    if return_address:
      write_address = torch.zeros(
        n_data,
        2,
        device=self.device,
        dtype=torch.int64
      )
      write_address[:, 0] = cells
    for cell_index in unique_cells:
      cell_index = cell_index.item()
      cell_size = self._cell_size[cell_index]
      cell_cap = self._storage[cell_index].shape[1]
      mask = cells == cell_index
      selected_data = data[:, mask] #[code_size, n_selected_data]
      selected_ids = ids[mask]
      n_selected_data = selected_data.shape[1]
      while cell_cap - cell_size < n_selected_data:
        self.expand_cell(cell_index)
        cell_cap = self._storage[cell_index].shape[1]
      self._storage[cell_index][:, cell_size : cell_size + n_selected_data] = self._reshape_data(selected_data)
      self._address2id[cell_index][cell_size : cell_size + n_selected_data] = selected_ids
      if return_address:
        write_address[mask, 1] = torch.arange(
          n_selected_data,
          device=self.device,
          dtype=torch.int64
        ) + cell_size
      self._cell_size[cell_index] += n_selected_data
    self._max_id = max(self._max_id, ids.max().item())
    n_unique_cells = unique_cells.shape[0]
    self.print_message(f"{n_data} new items added to {n_unique_cells} cells", 1)

    if return_address:
      return (ids, write_address)
    else:
      return ids

  def remove(self, ids=None, address=None):
    raise NotImplementedError
    # if ids is not None:
    #   address = self.get_address_by_id(ids)
    # elif address is not None:
    #   address = address.to(self.device)
    #   util.check_dtype(address, torch.int64)
    # else:
    #   raise RuntimeError("Need either ids or address")
    
    # mask = ()

  