#@title CellContainer
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
    return getattr(self, str(key))

  def __setitem__(self, key, value):
    buffer = getattr(self, str(key))
    buffer[:] = value

  def __iter__(self):
    for i in range(self.n_buffers):
      yield getattr(self, str(i))

  def get_ptrs(self):
    return [buffer.data_ptr() for buffer in self]

class CellContainer_v2(CustomModule):
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
      for i in range(n_cells)
    ]
    self._storage = BufferList(storage)

    cell_ptr = self.storage.get_ptrs()
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


  @property
  def capacity(self):
    return self._capacity

  @property
  def n_items(self):
    return self._n_items

  @property
  def max_id(self):
    return self._max_id

  def get_id_by_address(self, address):
    """
      address: torch.Tensor
        dtype : int32
        shape : [n_address, 2]
    """
    pass

  def get_address_by_id(self, ids):
    pass

  def get_data_by_address(self, address):
    pass

  def set_data_by_address(self, data, address):
    pass
  
  def create_inverse_id_mapping(self):
    pass

  def empty(self):
    pass

  def expand(self):
    pass

  def add(self):
    pass

  def remove(self):
    pass

  