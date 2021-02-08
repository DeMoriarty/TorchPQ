import torch
import torch.nn as nn

from kmeans import MultiKMeans

class PQ(nn.Module):
  def __init__(
      self,
      d_vector,
      n_subvectors=8,
      n_clusters=256,
      mode="euclidean",
      verbose=0
    ):
    super(PQ, self).__init__()
    self.n_subvectors = n_subvectors
    self.n_clusters = n_clusters
    self.d_vector = d_vector
    self.verbose = verbose
    assert d_vector % n_subvectors == 0

    self.d_subvector = d_vector // n_subvectors
    self.mode = mode
    
    self.kmeans = MultiKMeans(
      n_clusters = n_clusters,
      mode = mode,
      max_iter = 25,
      verbose = verbose,
    )

  @property
  def codebook(self):
    """
      return : torch.Tensor, 
        shape :[n_subvectors, d_subvectors, n_clusters]
    """
    return self.kmeans.centroids

  def train_codebook(self, x):
    """
      x: torch.Tensor, shape : [d_vector, n_data], dtype : float32
    """
    d_vector, n_data = x.shape
    assert d_vector == self.d_vector
    x = x.reshape(self.n_subvectors, self.d_subvector, n_data)
    return self.kmeans.fit(x)

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

  def decode(self, code):
    """
      code: torch.Tensor, shape : [n_subvectors, n_data], dtype : uint8
      return: torch.Tensor, shape : [d_vector, n_data], dtype : float32
    """
    n_subvectors, n_data = code.shape
    assert n_subvectors == self.n_subvectors
    arange = torch.arange(n_subvectors)[:, None].expand(-1, n_data)
    res = self.codebook[arange, :, code.long()]
    res = res.transpose(1, 2).reshape(self.d_vector, n_data)
    return res