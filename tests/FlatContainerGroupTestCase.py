import numpy as np
import yaml
import torch

class FlatContainerGroupTestCase(CustomTestCase):
  def setUp(self):
    doc = self.shortDescription()
    if (doc is None) or ("skip_test" not in doc):
      with open("configs/FlatContainerGroupConfig.yml", "r") as f:
        config = yaml.load(f)
      self.module = FlatContainerGroup(
        **config
      )
      self.device = self.module.device

  def tearDown(self):
    doc = self.shortDescription()
    if (doc is None) or ("skip_test" not in doc):
      del self.module

  def create_data(self, n_data):
    data_list = []
    for i in range(self.module.n_storage):
      dtype = self.module.dtype_list[i]
      device = self.module.device_list[i]
      code_size = self.module.code_size_list[i]
      if dtype in [torch.int32, torch.int64, torch.uint8]:
        data = torch.randint(
          2 ** 31,
          size = [code_size, n_data],
          device = device,
          dtype = dtype
        )
      else:
        data = torch.randn(
          code_size, n_data,
          device = device,
          dtype = dtype
        )
      data_list.append(data)
    return data_list

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

  def test_init(self):
    pass

  def test_expand(self):
    for i in range(8):
      cap_before = self.module.capacity
      with self.subTest(i = cap_before):
        self.module.expand()
        cap_after = self.module.capacity
        if self.module.expand_mode == "double":
          self.assertTrue(cap_after / cap_before == 2.0)
        elif self.module.expand_mode == "step":
          self.assertTrue(cap_after - cap_before == self.module.expand_step_size)

  def test_add_with_ids(self):
    n_data = 10000
    data_list = self.create_data(n_data)
    ids = self.create_ids_unique(n_data)
    returned_ids, returned_adr = self.module.add(
      data_list,
      ids=ids,
      return_address=True
    )
    self.assertTensorEqual(ids, returned_ids)

    adr = self.module.get_address_by_id(ids)
    self.assertTensorEqual(adr, returned_adr)

    returned_data_list = self.module.get_data_by_address(returned_adr)
    self.assertTrue(len(returned_data_list) == self.module.n_storage)
    for i in range(self.module.n_storage):
      with self.subTest(i = i):
        returned_data = returned_data_list[i]
        data = data_list[i]
        self.assertTensorEqual(data, returned_data)

  def test_add_without_ids(self):
    n_data = 10000
    data_list = self.create_data(n_data)
    returned_ids, returned_adr = self.module.add(
      data_list,
      return_address=True
    )
    ids = self.module.get_id_by_address(returned_adr)
    self.assertTensorEqual(ids, returned_ids)

    adr = self.module.get_address_by_id(returned_ids)
    self.assertTensorEqual(adr, returned_adr)

    returned_data_list = self.module.get_data_by_address(returned_adr)
    self.assertTrue(len(returned_data_list) == self.module.n_storage)
    for i in range(self.module.n_storage):
      with self.subTest(i = i):
        returned_data = returned_data_list[i]
        data = data_list[i]
        self.assertTensorEqual(data, returned_data)

  def test_remove_by_address(self):
    n_data = 10000
    data_list = self.create_data(n_data)
    ids = self.create_ids_unique(n_data)
    _, returned_adr = self.module.add(
      data_list,
      ids=ids,
      return_address=True
    )

    self.module.remove(address = returned_adr)

    # ???
    returned_ids = self.module.get_id_by_address(returned_adr)
    self.assertTensorEqual(returned_ids, -1)

  def test_remove_by_ids(self):
    n_data = 10000
    data_list = self.create_data(n_data)
    ids = self.create_ids_unique(n_data)
    _, returned_adr = self.module.add(
      data_list,
      ids=ids,
      return_address=True
    )

    self.module.remove(ids = ids)

    address = self.module.get_address_by_id(ids)
    self.assertTensorEqual(address, -1)
    
    # ???
    returned_ids = self.module.get_id_by_address(returned_adr)
    self.assertTensorEqual(returned_ids, -1)
    
  def test_add_remove_interaction(self):
    n_data = 1000
    data_list = self.create_data(n_data)
    ids = self.create_ids_unique(n_data)
    _, returned_adr = self.module.add(
      data_list,
      ids=ids,
      return_address=True
    )

    n_remove = n_data // 2
    remove_idx = np.random.choice(
      n_data,
      size=[n_remove],
      replace=False)
    remove_adr = returned_adr[remove_idx]
    self.module.remove(address = remove_adr)

    new_data_list = self.create_data(n_remove)
    new_ids, new_adr = self.module.add(
      new_data_list,
      return_address=True
    )

    returned_new_data_list = self.module.get_data_by_address(new_adr)
    for i in range(self.module.n_storage):
      with self.subTest(i = i):
        returned_new_data = returned_new_data_list[i]
        new_data = new_data_list[i]
        self.assertTensorEqual(new_data, returned_new_data)

    returned_new_ids = self.module.get_id_by_address(new_adr)
    self.assertTensorEqual(new_ids, returned_new_ids)

    returned_new_adr = self.module.get_address_by_id(new_ids)
    self.assertTensorEqual(new_adr, returned_new_adr)
    
  def test_empty(self):
    self.assertTrue(self.module.n_items == 0)
    self.assertTensorEqual(self.module._address2id >= 0, 0)
    