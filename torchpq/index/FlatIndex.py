import torch
import numpy as np
from ..container import FlatContainer
from ..fn import Topk
from .. import util
from .. import metric

class FlatIndex(FlatContainer):
  def __init__(
    self,
    d_vector,
    initial_size=None,
    expand_step_size=1024,
    expand_mode="double",
    device="cuda:0",
    distance="euclidean",
    verbose=0,
    ):
    super(FlatIndex, self).__init__(
      code_size = d_vector,
      contiguous_size = 1,
      dtype = "float32",
      device = device,
      initial_size = initial_size,
      expand_step_size = expand_step_size,
      expand_mode = expand_mode,
      use_inverse_id_mapping = True,
      verbose = verbose,
    )
    self.d_vector = d_vector
    if distance in ["euclidean", "l2"]:
      self.distance = "euclidean"
    elif distance in ["cosine", "angular"]:
      self.distance = "cosine"
    elif distance in ["inner", "dot"]:
      self.distance = "inner"
    elif distance in ["manhattan", "l1"]:
      raise NotImplementedError("currently manhattan distance is not supported")
    else:
      raise NotImplementedError(f"unknown distance metric: {distance}")

    if self.device_type == "cuda":
      self._topk = Topk()

  def search(self, x, k=1, return_address=False):
    """
      search K nearest neighbors of each vector in `x`
      parameters:
        x: torch.Tensor
          dtype : float32
          shape : [d_vector, n_query]
          batch of queries, needs to be on the same device as index

        k: int
          default : 1
          k in topk

        return_address: bool
          default: False
          if True, return (topk_values, topk_ids, topk_address)
          if False, return (topk_values, topk_ids)

      return:
        topk_values: torch.Tensor
          dtype : float32
          shape : [n_query, k]
          similarity scores of k nearest neighbors

        topk_ids: torch.Tensor
          dtype : int64
          shape : [n_query, k]
          ids of k nearest neighbors

        topk_address: torch.Tensor
          dtype : int64
          shape : [n_query, k]
          address of k nearest neighbors

    """
    d_vector, n_query = x.shape
    assert d_vector == self.d_vector
    assert util.check_dtype(x, "float32")
    assert util.check_device(x, self.device)
    assert k >= 1
    storage = self._storage.view(self.d_vector, -1) #[d_vector, n_data]
    if self.distance == "euclidean":
      sims = metric.negative_squared_l2_distance(x, storage, inplace=True)
    elif self.distance == "cosine":
      sims = metric.cosine_similarity(x, storage, normalize=True, inplace=True)
    elif self.distance == "inner":
      sims = metric.cosine_similarity(x, storage, normalize=False, inplace=True)

    if self.device_type == "cuda":
      topk_val, topk_address = self._topk(sims, k=k, dim=1)
    elif self.device_type == "cpu":
      topk_val, topk_address = torch.topk(sims, k=k, dim=1)

    topk_ids = self.get_id_by_address(topk_address)
    if return_address:
      return topk_val, topk_ids, topk_address
    else:
      return topk_val, topk_ids