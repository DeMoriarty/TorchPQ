import torch
import numpy as np
from ..CustomModule import CustomModule
class PCA(CustomModule):
  def __init__(self, n_components):
    """
      Principle Component Analysis (PCA)
      n_components: int
        number of principle components
    """
    super(PCA, self).__init__()
    assert n_components > 0
    self.n_components = n_components
    self.register_buffer("_mean", None)
    self.register_buffer("_components", None)
    
  @staticmethod
  def covar(x, meaned=True, rowvar=True, inplace=False):
    """
      compute covariance matrix of 'x'
      x: torch.Tensor, shape : [m, n]
      meaned: bool, default : True
        if True, assume 'x' has zero mean
      rowvar: bool, default : True
        if True, assume 'm' represents n_features and 'n' represents n_samples
        if False, assume 'm' represents n_samples and 'n' represents n_features
      inplace: bool, default : False
        if meaned is False and inplace is True, mean of 'x' will be subtracted from 'x' inplace,
        and will be added back to 'x' at the end, this will prevent creating a new tensor of shape [m, n]
        with the cost of extra computation.
    """
    if x.dim() > 2:
      raise ValueError('x has more than 2 dimensions')
    if x.dim() < 2:
      x = x.view(1, -1)
    if not rowvar and x.shape[0] != 1:
      x = x.T
    fact = 1.0 / (x.shape[1] - 1)

    if not meaned:
      mean = x.mean(dim=1, keepdim=True)
      if inplace:
        x.sub_(mean)
      else:
        x = x - mean

    result = fact * (x @ x.T).squeeze()
    if inplace and not meaned:
      x.add_(mean)
    return result
  
  def train(self, x, inplace=False):
    """
      train PCA with 'x'
      x: torch.Tensor, shape : [d_vec, n_sample]
      inplace: bool, default : False
        if True, reduce the memory consumption with the cost of extra computation
    """
    assert x.shape[0] >= self.n_components

    mean = x.mean(dim=1, keepdim=True) #[d_vec, 1]
    if inplace:
      x.sub_(mean)
    else:
      x = x - mean
    x_cov = self.covar(x, rowvar=True, meaned=True)
    if inplace:
      x.add_(mean)
    eig_val, eig_vec = torch.symeig(x_cov, eigenvectors=True, upper=False)
    sorted_eig_val, sorted_index = eig_val.sort(descending=True)
    sorted_eig_vec = eig_vec[:, sorted_index]
    components = sorted_eig_vec[:, :self.n_components].T
    self.register_buffer("_components", components)
    self.register_buffer("_mean", mean)

  def encode(self, x):
    """
      reduce the dimentionality of 'x' from 'd_vec' to 'n_components' 
      x: torch.Tensor, shape : [d_vec, n_samples], dtype : float32
      return: torch.Tensor, shape : [n_components, n_samples], dtype : float32
    """
    assert self._components is not None
    assert x.shape[0] == self._components.shape[1]

    x = x - self._mean
    y = self._components @ x
    return y
  
  def decode(self, x):
    """
      reconstruct 'x' from 'n_components' dimentional space to 'd_vec' dimentional space 
      x: torch.Tensor, shape : [n_components, n_samples], dtype : float32
      return: torch.Tensor, shape : [d_vec, n_samples], dtype : float32
    """
    assert self._components is not None
    assert x.shape[0] == self._components.shape[0]

    y = self._components.T @ x
    y = y + self._mean
    return y