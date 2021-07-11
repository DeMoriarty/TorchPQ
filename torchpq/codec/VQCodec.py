import torch

from ..clustering import KMeans
# from ..clustering import KMeansOld as KMeans
from .BaseCodec import BaseCodec

class VQCodec(BaseCodec):
  def __init__(self, *args, **kwargs):
    super(VQCodec, self).__init__()
    self.kmeans = KMeans(
      *args,
      **kwargs
    )

  @property
  def codebook(self):
    return self.kmeans.centroids

  def encode(self, input):
    """
      input:
        torch.Tensor
        shape : [d_vector, n_data]
        dtype : float32

      returns:
        torch.Tensor
        shape : [n_data]
        dtype : int64
    """
    assert self._is_trained == True, "codec is not trained"
    return self.kmeans.predict(input)

  def decode(self, code):
    """
      code:
        torch.Tensor
        shape : [n_code]
        dtype : int64

      returns:
        torch.Tensor
        shape : [d_vector, n_data]
        dtype : float32
    """
    assert self._is_trained == True, "Codec is untrained"
    return self.codebook[:, code].clone()

  def train(self, data):
    """
      data:
        torch.Tensor
        shape : [d_vector, n_data]
        dtype : float32
    """
    labels = self.kmeans.fit(data)
    self._trained(True)
    return labels