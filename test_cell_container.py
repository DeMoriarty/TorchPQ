import torch
from torchpq.container import DistributedCellContainer

code_size = 32
n_cells = 16
dtype = torch.float32
# test initialization
container = DistributedCellContainer(
  code_size=code_size,
  n_cells=n_cells,
  contiguous_size=4,
  dtype=dtype,
  device="cuda",
  expand_step_size=32,
  expand_mode="double",
  use_inverse_id_mapping=True,
  verbose=0
)


# test add
n_data = 100
data = torch.randn(code_size, n_data, dtype=dtype, device="cuda:0")
cells = torch.randint(n_cells, size=[n_data], dtype=torch.long)
returned_ids, returned_address = container.add(data=data, cells=cells, return_address=True)

returned_data = container.get_data_by_address(returned_address)
dif = (data - returned_data).abs()
print("err1", dif.sum())

# test expand
cells_to_expand = [1, 3, 5, 7]
print(container._cell_capacity)
container.expand(cells=cells_to_expand)
print(container._cell_capacity)

# test get_address_by_id
returned_address2 = container.get_address_by_id(returned_ids)
returned_data = container.get_data_by_address(returned_address)
dif = (data - returned_data).abs()
print("err2", dif.sum())

# test get_id_by_address
returned_ids2 = container.get_id_by_address(returned_address)
dif = returned_ids2 != returned_ids
print("err3", dif.sum())
