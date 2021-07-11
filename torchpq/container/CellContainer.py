#@title CellContainer
import torch
from .. import util
from ..kernels import GetDivByAddressV2Cuda
from ..kernels import GetIOACuda
from ..kernels import GetWriteAddressV2Cuda
from .BaseContainer import BaseContainer

class CellContainer(BaseContainer):
  def __init__(
      self,
      code_size,
      n_cells,
      dtype="float32",
      device="cpu",
      initial_size=None,
      expand_step_size=1024,
      expand_mode="double",
      use_inverse_id_mapping=False,
      contiguous_size=1,
      verbose=0
    ):
    if initial_size is None:
      initial_size = expand_step_size

    super(CellContainer, self).__init__(
      device = device,
      initial_size = initial_size * n_cells,
      expand_step_size = expand_step_size,
      expand_mode = expand_mode,
      use_inverse_id_mapping = use_inverse_id_mapping
    )
    assert n_cells > 0
    assert code_size > 0
    assert code_size % contiguous_size == 0
    if type(dtype) == str:
      dtype = util.str2dtype(dtype)
    self.n_cells = n_cells
    self.code_size = code_size
    self.dtype = dtype
    self.contiguous_size = contiguous_size
    self.initial_size = initial_size
    self.verbose = verbose

    _storage = torch.zeros(
      code_size // contiguous_size,
      n_cells * initial_size,
      contiguous_size,
      device = device,
      dtype = dtype
    )
    self.register_buffer("_storage", _storage)

    _cell_start = torch.arange(
      n_cells,
      device = device
    ) * initial_size
    self.register_buffer("_cell_start", _cell_start)

    _cell_size = torch.zeros(
      n_cells,
      device = device,
      dtype = torch.long
    )
    self.register_buffer("_cell_size", _cell_size)

    _cell_capacity = torch.zeros(
      n_cells,
      device = device,
      dtype = torch.long
    ) + initial_size
    self.register_buffer("_cell_capacity", _cell_capacity)

    _is_empty = torch.ones(
      n_cells * initial_size,
      device = device,
      dtype=torch.uint8
    )
    self.register_buffer("_is_empty", _is_empty)

    self._get_cell_by_address_cuda = GetDivByAddressV2Cuda(
      ta=4,
      tpb=256,
    )
    self._get_ioa_cuda = GetIOACuda(
      tpb=256,
    )
    self._get_write_address_cuda = GetWriteAddressV2Cuda(
      tpb=256,
    )

  @property
  def n_items(self):
    return self._cell_size.sum().item()

  def _get_cell_by_address_cpu(self, address, cell_start, cell_end):
    n_address = address.shape[0]
    mask1 = cell_start[None, ] <= address[:, None]
    mask2 = cell_end[None, ] > address[:, None] # [n_address, n_cq_clusters]
    mask = mask1 & mask2
    not_found = mask.sum(dim=1) == 0
    mask[not_found, 0] = True
    cells = torch.nonzero(mask)
    cells[not_found, 1] = -1
    return cells[:, 1]

  def get_cell_by_address(self, address):
    assert util.check_dtype(address, torch.int64)
    address = address.to(self.device)
    cell_start = self._cell_start
    cell_end = cell_start + self._cell_capacity
    if address.device.type == "cuda":
      return self._get_cell_by_address_cuda(address, cell_start, cell_end)
    elif address.device.type == "cpu":
      return self._get_cell_by_address_cpu(address, cell_start, cell_end)

  def _get_ioa_cpu(self, cells, unique_cells=None):
    if unique_cells is None:
      unique_cells = torch.unique(cells) #[n_unique_clusters]
    expanded_cells = cells[:, None].expand(-1, unique_cells.shape[0]) #[n_data, n_unique_clusters]
    mask = expanded_cells == unique_cells[None, :] #[n_data, n_unique_clusters]
    mcs = mask.cumsum(dim=0)
    mcs[[~mask]] = 0
    ioa = mcs.sum(dim=1) - 1
    return ioa

  def get_ioa(self, cells, unique_cells=None):
    assert util.check_dtype(cells, torch.int64)
    cells = cells.to(self.device)
    if unique_cells is not None:
      assert util.check_dtype(unique_cells, torch.int64)
      unique_cells = unique_cells.to(self.device)

    if cells.device.type == "cuda":
      return self._get_ioa_cuda(cells, unique_cells)
    elif cells.device.type == "cpu":
      return self._get_ioa_cpu(cells, unique_cells)

  def _get_write_address_cpu(self, empty_adr, cells, ioa):
    n_cells = cells.shape[0]
    cell_of_empty_adr = self.get_cell_by_address(empty_adr)
    write_address = torch.zeros_like(cells)
    for i in range(n_cells):
      cell_mask = cell_of_empty_adr == cells[i]
      write_adr = empty_adr[cell_mask]
      write_adr = write_adr[ioa[i] ]
      write_address[i] = write_adr
    return write_address

  def get_write_address(self, cells, empty_adr=None, ioa=None):
    assert util.check_dtype(cells, torch.int64)
    cells = cells.to(self.device)
    if empty_adr is not None:
      assert util.check_dtype(empty_adr, torch.int64)
      empty_adr = empty_adr.to(self.device)

    if ioa is not None:
      assert util.check_dtype(ioa, torch.int64)
      ioa = ioa.to(self.device)
    else:
      ioa = self.get_ioa(cells)

    if cells.device.type == "cuda":
      return self._get_write_address_cuda(
        self._is_empty,
        self._cell_start,
        self._cell_capacity,
        cells,
        ioa)
    elif cells.device.type == "cpu":
      assert empty_adr is not None, "empty_adr is required when device is cpu"
      return self._get_write_address_cpu(empty_adr, cells, ioa)

  def get_data_by_address(self, address):
    """
    address:
      torch.Tensor,
      shape : [n_data],
      dtype : int64
    
    returns data:
      torch.Tensor,
      shape : [code_size, n_data],
      dtype : self.dtype

    """
    assert util.check_dtype(address, torch.int64)
    address = address.to(self.device)

    n_address = address.shape[0]
    mask = (0 <= address) & (address < self.capacity)
    address[~mask] = 0
    data = self._storage.index_select(
      dim = 1,
      index = address
    )
    data.index_fill_(
      dim = 1,
      index = torch.nonzero(~mask).squeeze(1),
      value = 0,
    )
    # data = torch.zeros(
    #   self.code_size // self.contiguous_size,
    #   n_address,
    #   contiguous_size,
    #   device = self.device
    # )
    # data[:, mask] = self._storage[:, address[mask]]
    data = data.transpose(1, 2).reshape(self.code_size, -1)
    return data

  def set_data_by_address(self, data, address):
    """
    data:
      torch.Tensor,
      shape : [code_size, n_data]
      dtype : self.dtype

    address:
      torch.Tensor,
      shape : [n_data],
      dtype : int64
    """
    assert util.check_dtype(address, torch.int64)
    assert util.check_dtype(data, self.dtype)
    assert data.shape[0] == self.code_size
    assert data.shape[1] == address.shape[0]
    n_address = address.shape[0]
    address = address.to(self.device)
    data = data.to(self.device)
    data = data.reshape(
      self.code_size // self.contiguous_size,
      self.contiguous_size,
      -1
    ).transpose(1, 2)

    mask = (0 <= address) & (address < self.capacity)
    self._storage[:, address[mask]] = data[:, mask]

  def empty(self):
    super(CellContainer, self).empty()
    self._storage.fill_(0)
    self._cell_size.fill_(0)
    self._is_empty.fill_(1)
    self._n_items = 0
    self.print_message("index has been empties", 2)

  def __expand_v2(self):
    storage = self._storage
    address2id = self._address2id
    is_empty = self._is_empty
    del self._storage, self._address2id, self._is_empty
    selected_cell_size = self._cell_size[cells]
    if self.expand_mode == "step":
      n_new = cells.shape[0] * self.expand_step_size
    elif self.expand_mode == "double":
      n_new = selected_cell_size.sum().item()

    new_storage = torch.zeros(
      self.code_size // self.contiguous_size,
      storage.shape[1] + n_new,
      self.contiguous_size,
      device = self.device,
      dtype = self.dtype
    )
    new_a2i = torch.zeros(
      storage.shape[1] + n_new,
      device = self.device,
      dtype = torch.long
    ) - 1
    new_is_empty = torch.ones(
      storage.shape[1] + n_new,
      device = self.device,
      dtype = torch.uint8
    )

    map_to_new = torch.arange(
      storage.shape[1],
      device=self.device,
      dtype=torch.long
    )
    
    old_cell_start = self._cell_start.clone()
    old_cell_capacity = self._cell_capacity.clone()
    for cell_index in cells:
      if self.expand_mode == "step":
        cell_n_new = self.expand_step_size
      elif self.expand_mode == "double":
        cell_n_new = cell_cap
      cell_start = old_cell_start[cell_index].item()
      cell_cap = old_call_capacity[cell_index].item()
      cell_end = cell_start + cell_cap
      new_cell_start = self._cell_start[cell_index].item()
      new_cell_cap = self._cell_capacity[cell_index].item()

      map_to_new[cell_start : cell_end] += - cell_start + new_cell_start

      self._call_capacity[cell_index] += cell_n_new
      arange = torch.arange(
        start = cell_index+1,
        end=self.n_cells
      )
      self._cell_start[arange] += cell_n_new
    new_storage[:, map_to_new] = storage
    new_a2i[map_to_new] = address2id
    new_is_empty[map_to_new] = is_empty

  def expand(self, cells):
    n_cells = cells.shape[0]
    storage = self._storage
    address2id = self._address2id
    is_empty = self._is_empty
    del self._storage, self._address2id, self._is_empty

    total = 0
    for cell_index in cells:
      cell_start = self._cell_start[cell_index].item()
      cell_cap = self._cell_capacity[cell_index].item()
      cell_end = cell_start + cell_cap
      if self.expand_mode == "step":
        n_new = self.expand_step_size
      elif self.expand_mode == "double":
        n_new = cell_cap

      new_block = torch.zeros(
        self.code_size // self.contiguous_size,
        n_new,
        self.contiguous_size,
        device = self.device,
        dtype = self.dtype
      )
      storage = torch.cat([
        storage[:, :cell_end],
        new_block,
        storage[:, cell_end:]
      ], dim=1)

      new_a2i = torch.zeros(
        n_new,
        device = self.device,
        dtype = torch.int64
      ) - 1
      address2id = torch.cat([
        address2id[:cell_end],
        new_a2i,
        address2id[cell_end:]
      ], dim=0)

      new_is_empty = torch.ones(
        n_new,
        device = self.device,
        dtype=torch.uint8,
      )
      is_empty = torch.cat([
        is_empty[:cell_end],
        new_is_empty,
        is_empty[cell_end:]
      ], dim=0)

      self._cell_capacity[cell_index] += n_new
      arange = torch.arange(
        start = cell_index+1,
        end=self.n_cells
      )
      self._cell_start[arange] += n_new
      total += n_new
    self.register_buffer("_storage", storage)
    self.register_buffer("_address2id", address2id)
    self.register_buffer("_is_empty", is_empty)
    self.print_message(f"Total storage capacity is expanded by {total} for {n_cells} cells", 2)

  def add(self, data, cells, ids=None, return_address=False):
    assert util.check_dtype(data, self.dtype)
    assert util.check_dtype(cells, torch.long)
    assert data.shape[0] == self.code_size
    assert data.shape[1] == cells.shape[0]
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
    ioa = self.get_ioa(cells, unique_cells)

    # expand storage if necessary
    while True:
      free_space = self._cell_capacity[cells] - self._cell_size[cells] - (ioa + 1)
      expansion_required = cells[free_space < 0].unique()
      # print("exp req", expansion_required)
      if expansion_required.shape[0] == 0:
        break
      self.expand(expansion_required)

    #get write address
    empty_adr = torch.nonzero(self._is_empty == 1)[:, 0] #[n_empty]
    write_address = self.get_write_address(
      empty_adr = empty_adr,
      cells = cells,
      ioa = ioa
    )
    self.set_data_by_address(data, write_address)

    self._address2id[write_address] = ids
    self._max_id = max(self._max_id, ids.max().item())
    self._is_empty[write_address] = 0

    # update number of stored items in each cell
    self._cell_size[unique_cells] += unique_cell_counts

    self.print_message(f"{n_data} new items added", 1)

    if return_address:
      return (ids, write_address)
    else:
      return ids

  def remove(self, ids=None, address=None):
    if ids is not None:
      address = self.get_address_by_id(ids)
    elif address is not None:
      address = address.to(self.device)
      util.check_dtype(address, torch.int64)
    else:
      raise RuntimeError("Need either ids or address")

    mask = (address >= 0) & (address < self.capacity)
    address = address[mask].unique(sorted=True)
    n_removed = address.shape[0]
    if n_removed <= self.n_items:
      self.print_message("no enough items in index to be removed", 1)
      return
    if n_removed == 0:
      return

    self._is_empty[address] = 1
    self._address2id[address] = -1

    cells = self.get_cell_by_address(address)
    unique_cells, counts = cells.unique(return_counts = True)
    self._cell_size[unique_cells] -= counts
    self.print_message(f"{n_removed} items has been removed", 2)
