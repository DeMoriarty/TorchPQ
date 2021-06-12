import torch
import torch.nn as nn

from .kmeans import MultiKMeans
from .kernels import PQDecodeCUDA
from .CustomModule import CustomModule

class PQ(CustomModule):
  def __init__(
      self,
      d_vector,
      n_subvectors=8,
      n_clusters=256,
      distance="euclidean",
      verbose=0
    ):
    super(PQ, self).__init__()
    self.n_subvectors = n_subvectors
    self.n_clusters = n_clusters
    self.d_vector = d_vector
    self.verbose = verbose
    assert d_vector % n_subvectors == 0

    self.d_subvector = d_vector // n_subvectors
    self.distance = distance
    
    self.kmeans = MultiKMeans(
      n_clusters = n_clusters,
      distance = distance,
      max_iter = 25,
      verbose = verbose,
    )

    self._decode_cuda = PQDecodeCUDA(tm=2, td=8)

  @property
  def codebook(self):
    """
      return : torch.Tensor, 
        shape :[n_subvectors, d_subvectors, n_clusters]
    """
    return self.kmeans.centroids

  def train(self, x):
    """
      x: torch.Tensor, shape : [d_vector, n_data], dtype : float32
    """
    d_vector, n_data = x.shape
    assert d_vector == self.d_vector
    x = x.reshape(self.n_subvectors, self.d_subvector, n_data)
    return self.kmeans.fit(x)

  def precompute_adc(self, query):
    """
      x: torch.Tensor, shape : [d_vector, n_query], dtype : float32
    """
    n_query = query.shape[1]
    assert query.shape[0] == self.d_vector
    
    query = query.reshape(self.n_subvectors, self.d_subvector, n_query)
    precomputed = self.kmeans.sim(query, self.codebook, normalize=False)
    return precomputed

  def encode(self, x):
    """
      x: torch.Tensor, shape : [d_vector, n_data], dtype : float32
      return: torch.Tensor, shape : [n_subvectors, n_data], dtype : uint8
    """
    d_vector, n_data = x.shape
    assert d_vector == self.d_vector
    x = x.reshape(self.n_subvectors, self.d_subvector, n_data)
    _, labels = self.kmeans.get_labels(x, self.codebook) #[n_subvectors, n_data]
    return labels.byte()

  @staticmethod
  def _decode_cpu(codebook, code):
    """
      code: torch.Tensor, shape : [n_subvectors, n_data], dtype : uint8
      return: torch.Tensor, shape : [d_vector, n_data], dtype : float32
    """
    n_subvectors, n_data = code.shape
    arange = torch.arange(n_subvectors)[:, None].expand(-1, n_data)
    res = codebook[arange, :, code.long()]
    res = res.transpose(1, 2).reshape(-1, n_data)
    return res


  def decode(self, code):
    """
      code: torch.Tensor, shape : [n_subvectors, n_data], dtype : uint8
      return: torch.Tensor, shape : [d_vector, n_data], dtype : float32
    """
    assert code.shape[0] == self.n_subvectors
    if self.codebook.device.type == "cpu":
      return self._decode_cpu(self.codebook, code)
    elif self.codebook.device.type == "cuda":
      return self._decode_cuda(self.codebook, code)