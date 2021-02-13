from time import time
import torch
import numpy as np
import torchpq

_iters = 1
class TestIVFPQR:
  def __init__(self, model):
    assert type(model).__name__ == "IVFPQR"
    self.model = model

  @staticmethod
  def timer(func, name="", *args, **kwargs):
    global _iters
    if torch.cuda.is_available():
      sync = torch.cuda.synchronize
    else:
      def do_nothing(): return None
      sync = do_nothing

    tm = time()
    for i in range(_iters):
      result = func(*args, **kwargs)
    sync(); print(name, (time()-tm) / _iters)
    return result

  def print_shape(self):
    print("shape of model components:")
    print("---storage", self.model.storage.shape, self.model.storage.dtype)
    print("---div_start", self.model.div_start.shape, self.model.div_start.dtype)
    print("---div_capacity", self.model.div_capacity.shape, self.model.div_capacity.dtype)
    print("---div_size", self.model.div_size.shape, self.model.div_size.dtype)
    print("---ids", self.model.ids.shape, self.model.ids.dtype)
    print("---locations", self.model.locations.shape, self.model.locations.dtype)
    if self.model.coarse_q.centroids is not None:
      print("---cq_centroids", self.model.coarse_q.centroids.shape, self.model.coarse_q.centroids.dtype)
    if self.model.product_q.codebook is not None:
      print("---pq_centroids", self.model.product_q.codebook.shape, self.model.product_q.codebook.dtype)
    print("---model", self.model)
 
  def test_train(self, data):
    return self.timer(self.model.train, "train", data)

  def test_encode(self, query):
    return self.timer(self.model.encode, "encode", query)

  def test_decode(self, code, query=None):
    recon = self.timer(self.model.decode, "decode", code)
    if query is not None:
     dif1 = query - recon
    dif2 = query - torch.randn_like(query)
    dif1.pow_(2)
    dif2.pow_(2)
    print("reconstruction error:", dif1.sum(dim=0).sqrt().mean() )
    print("random error:", dif2.sum(dim=0).sqrt().mean())
    return recon

  def test_expand(self, divs):
    return self.timer(self.model.expand, "expand", divs)

  def test_add(self, input, input_ids=None):
    return self.timer(self.model.add, "add", input, input_ids)

  def test_remove(self, ids):
    return self.timer(self.model.remove, "remove", ids)

  def test_get_address_of_id(self, ids):
    func = self.model.get_address_of_id
    name = "get_address_of_id"
    return self.timer(func, name, ids)

  def test_get_id_of_address(self, address, ids=None):
    func = self.model.get_id_of_address
    name = "get_id_of_address"
    rids = self.timer(func, name, address)
    if ids is not None:
      print("id recon error", (ids != rids).sum() )
    return rids

  def test_get_div_of_address(self, address, divs=None):
    func = self.model.get_div_of_address
    name = "get_div_of_address"
    rdivs = self.timer(func, name, address)
    if divs is not None:
      print("div recon error", (divs != rdivs).sum() )
    return rdivs
  
  def test_get_data_of_address(self, address):
    func = self.model.get_data_of_address
    name = "get_data_of_address"
    return self.timer(func, name, address)

  def test_get_data_of_id(self, ids, code=None):
    func = self.model.get_data_of_id
    name = "get_data_of_id"
    result = self.timer(func, name, ids)
    if code is not None:
      print("get_data_of_id error", (result != code).sum())
    return result

  def test_topk(self, query, k, mode=2):
    func = self.model.topk
    name = "topk"
    return self.timer(func, name, query, k, mode)

  def test_repr(self):
    print("__repr__")
    print(self.model)

  def test_len(self):
    print("__len__")
    print(len(self.model))
    return len(self.model)

  def test_getitem(self, ids):
    func = self.model.__getitem__
    name = "__getitem__"
    return self.timer(func, name, ids)

  def test_all(self, data, query, ids=None, divs_to_expand=None):
    self.test_train(data)
    code = self.test_encode(query)
    recon = self.test_decode(code, query)
    
    if divs_to_expand is not None:
      self.test_expand(divs_to_expand)

    ids = self.test_add(query, ids)
    self.test_repr()
    self.test_len()
    self.test_getitem(ids)

    address = self.test_get_address_of_id(ids)
    rid = self.test_get_id_of_address(address, ids)
    divs = self.model.coarse_q.predict(query)
    rdiv = self.test_get_div_of_address(address, divs)
    data = self.test_get_data_of_id(ids, code)
    self.test_remove(ids)