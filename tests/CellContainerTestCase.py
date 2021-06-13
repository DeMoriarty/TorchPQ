import numpy as np
import torch
import yaml
from torchpq.container import CellContainer

class CellContainerTestCase(CustomTestCase):
  def setUp(self):
    doc = self.shortDescription()
    if (doc is None) or ("skip_test" not in doc):
      with open("configs/CellContainerConfig.yml", "r") as f:
        config = yaml.load(f)
      self.module = CellContainer(
        **config
      )
      self.device = self.module.device
      self.dtype = self.module.dtype

  def tearDown(self):
    doc = self.shortDescription()
    if (doc is None) or ("skip_test" not in doc):
      del self.module

  def create_data_randn(self, n_data):
    return torch.randn(
      self.module.code_size,
      n_data,
      device = self.module.device
    ).to(self.module.dtype)

  def create_data_rand(self, n_data):
    return torch.rand(
      self.module.code_size,
      n_data,
      device = self.module.device
    ).to(self.module.dtype)

  def create_data_randint(self, n_data, low=0, high=2**31):
    return torch.randint(
      low,
      high,
      size = [self.module.code_size, n_data],
      device = self.module.device,
    ).to(self.module.dtype)

  def create_address(self, n_address):
    return torch.randint(
      self.module.capacity,
      size = [n_address],
      device = self.module.device,
      dtype = torch.long,
    )

  def create_unique_address(self, n_address):
    address = np.random.choice(
      self.module.capacity,
      size = [n_address],
      replace = False,
    )
    return torch.tensor(cells, device=self.module.device).long()

  def create_ids_unique(self, n_ids):
    ids = np.random.randint(
      2**62,
      size = [n_ids],
    )
    ids = torch.tensor(ids, device=self.module.device).long()
    return ids

  def create_ids_arange(self, n_ids):
    return torch.arange(
      n_ids,
      device = self.module.device,
      dtype = torch.long,
    ) + self.module.max_id + 1

  def create_cells(self, n_cells):
    return torch.randint(
      self.module.n_cells,
      size = [n_cells],
      device = self.module.device,
      dtype = torch.long
    )
  
  def create_unique_cells(self, n_cells):
    cells = np.random.choice(
      self.module.n_cells,
      size = [n_cells],
      replace = False,
    )
    return torch.tensor(cells, device=self.module.device).long()

  def test_init(self):
    pass

  def test_expand(self):
    n_cells = self.module.n_cells
    n_expanded_cells = torch.randint(n_cells, size=[1]).item()
    expanded_cells = self.create_unique_cells(n_expanded_cells)
    old_cap = self.module._cell_capacity[expanded_cells]
    self.module.expand(expanded_cells)
    new_cap = self.module._cell_capacity[expanded_cells]
    if self.module.expand_mode == "double":
      self.assertErrorLessThan(new_cap, old_cap * 2)
    elif self.module.expand_mode == "step":
      self.assertErrorLessThan(new_cap, old_cap - self.module.expand_step_size)

  def test_add_with_ids(self):
    n_data = 10000
    data = self.create_data_randn(n_data)
    cells = self.create_cells(n_data)
    ids = self.create_ids_unique(n_data)
    returned_ids, returned_adr = self.module.add(
      data,
      cells=cells,
      ids=ids,
      return_address=True
    )
    self.assertTensorEqual(ids, returned_ids)

    adr = self.module.get_address_by_id(ids)
    self.assertTensorEqual(adr, returned_adr)

    returned_data = self.module.get_data_by_address(returned_adr)
    self.assertTensorEqual(data, returned_data)

    is_empty = self.module._is_empty[returned_adr]
    self.assertTensorEqual(is_empty, 0)

  def test_add_without_ids(self):
    n_data = 10000
    data = self.create_data_randn(n_data)
    cells = self.create_cells(n_data)
    returned_ids, returned_adr = self.module.add(
      data,
      cells=cells,
      return_address=True
    )
    ids = self.module.get_id_by_address(returned_adr)
    self.assertTensorEqual(ids, returned_ids)

    adr = self.module.get_address_by_id(returned_ids)
    self.assertTensorEqual(adr, returned_adr)

    returned_data = self.module.get_data_by_address(returned_adr)
    self.assertTensorEqual(data, returned_data)

    is_empty = self.module._is_empty[returned_adr]
    self.assertTensorEqual(is_empty, 0)

  def test_remove_by_address(self):
    n_data = 10000
    data = self.create_data_randn(n_data)
    cells = self.create_cells(n_data)
    ids = self.create_ids_unique(n_data)
    _, returned_adr = self.module.add(
      data,
      cells=cells,
      ids=ids,
      return_address=True
    )

    self.module.remove(address = returned_adr)

    returned_ids = self.module.get_id_by_address(returned_adr)
    self.assertTensorEqual(returned_ids, -1)

    is_empty = self.module._is_empty[returned_adr]
    self.assertTensorEqual(is_empty, 1)

  def test_remove_by_ids(self):
    n_data = 10000
    data = self.create_data_randn(n_data)
    cells = self.create_cells(n_data)
    ids = self.create_ids_unique(n_data)
    _, returned_adr = self.module.add(
      data,
      cells=cells,
      ids=ids,
      return_address=True
    )

    self.module.remove(ids = ids)

    is_empty = self.module._is_empty[returned_adr]
    self.assertTensorEqual(is_empty, 1)

    address = self.module.get_address_by_id(ids)
    self.assertTensorEqual(address, -1)
    
    returned_ids = self.module.get_id_by_address(returned_adr)
    self.assertTensorEqual(returned_ids, -1)
    
  def test_add_remove_interaction(self):
    n_data = 1000
    data = self.create_data_randn(n_data)
    cells = self.create_cells(n_data)
    ids = self.create_ids_unique(n_data)
    _, returned_adr = self.module.add(
      data,
      cells=cells,
      ids=ids,
      return_address=True
    )
    # plt.vlines(self.module._cell_start.cpu(), ymin=0, ymax=1)
    # plt.plot(self.module._is_empty.cpu(), ",")
    # plt.show()

    n_remove = n_data // 2
    remove_idx = np.random.choice(
      n_data,
      size=[n_remove],
      replace=False)
    remove_adr = returned_adr[remove_idx]
    self.module.remove(address = remove_adr)

    # plt.vlines(self.module._cell_start.cpu(), ymin=0, ymax=1)
    # plt.plot(self.module._is_empty.cpu(), ",")
    # plt.show()

    new_data = self.create_data_randn(n_remove)
    new_cells = self.create_cells(n_remove)
    new_ids, new_adr = self.module.add(
      new_data,
      cells=new_cells,
      return_address=True
    )
    # plt.vlines(self.module._cell_start.cpu(), ymin=0, ymax=1)
    # plt.plot(self.module._is_empty.cpu(), ",")
    # plt.show()
    returned_new_data = self.module.get_data_by_address(new_adr)
    self.assertTensorEqual(new_data, returned_new_data)

    returned_new_ids = self.module.get_id_by_address(new_adr)
    self.assertTensorEqual(new_ids, returned_new_ids)

    returned_new_adr = self.module.get_address_by_id(new_ids)
    self.assertTensorEqual(new_adr, returned_new_adr)
    
  def test_empty(self):
    self.assertTrue(self.module.n_items == 0)
    self.assertTensorEqual(self.module._cell_size, 0)
