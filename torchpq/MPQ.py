import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .kmeans import MultiKMeans
from .kmeans import KMeans
from .kernels import PQDecodeCUDA
from .PQ import PQ

class MPQ(nn.Module):
  def __init__(
    self,
    d_vector,
    n_subvectors=8,
    n_clusters=256,
    distance="euclidean",
    verbose=0,
    n_codebooks=64,
    ):
    super(MPQ, self).__init__()
    assert d_vector % n_subvectors == 0
    self.n_codebooks = n_codebooks
    self.d_vector = d_vector
    self.n_subvectors = n_subvectors
    self.d_subvector = d_vector // n_subvectors
    self.n_clusters = n_clusters
    self.distance = distance
    self.verbose = verbose
    self.group_size=512


    #codebook: [n_codebooks, n_subvectors, d_subvectors, n_clusters]
    self.register_buffer("codebook", None)

    self.kmeans = MultiKMeans(
      n_clusters = n_clusters,
      distance = distance,
      max_iter = 25,
      verbose = verbose,
    )

    self.codebook_selector = KMeans(
      n_clusters = n_codebooks,
      distance = distance,
      max_iter = 25,
      verbose = verbose,
    )

    self._decode_cuda = PQDecodeCUDA(tm=2, td=8)

  def train(self, x):
    """
      x: shape: [d_vector, n_data]
    """
    labels = self.codebook_selector.fit(x)

    # print("labels", labels.shape, labels.unique().shape )
    unique_labels = labels.unique()
    codebook = torch.zeros(
      self.n_codebooks,
      self.n_subvectors,
      self.d_subvector,
      self.n_clusters,
      device=x.device,
      dtype=torch.float32
    )

    for label in unique_labels:
      mask = labels == label
      sub_x = (
        x[:, mask]
        .reshape(self.n_subvectors, self.d_subvector, -1)
        .contiguous()
      )
      self.kmeans.fit(sub_x)
      codebook[label] = self.kmeans.centroids
    del self.codebook
    self.register_buffer("codebook", codebook)

  def encode(self, x):
    """
      returns code and codebook_index
      x: shape: [d_vector, n_data]
    """
    n_data = x.shape[1]

    labels = self.codebook_selector.predict(x)
    unique_labels, counts = labels.unique(return_counts=True)
    n_unique = unique_labels.shape[0]
    code = torch.zeros(self.n_subvectors, n_data, dtype=torch.uint8, device=self.codebook.device)
    for i in range(n_unique):
      label = unique_labels[i]
      mask = labels == label
      sub_x = (
        x[:, mask]
        .reshape(self.n_subvectors, self.d_subvector, -1)
        .contiguous()
      )
      sub_codebook = self.codebook[label].contiguous()
      _, sub_code = self.kmeans.get_labels(sub_x, sub_codebook)
      code[:, mask] = sub_code.byte()
    return (code, labels)

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

  def decode(self, code, codebook_index):
    """
      returns reconstruction of code
      code: [n_subvectors, n_data]
      codebook_index: shape : [n_data], dtype : uint8
    """
    n_data = code.shape[1]
    unique_labels, counts = codebook_index.unique(return_counts=True)
    recon = torch.zeros(
      self.d_vector,
      n_data,
      device=self.codebook.device,
      dtype=torch.float32,
    )
    for label in unique_labels:
      mask = codebook_index == label
      sub_code = code[:, mask].contiguous()
      sub_codebook = self.codebook[label].contiguous()
      if self.codebook.device.type == "cpu":
        sub_recon = self._decode_cpu(sub_codebook, sub_code)
      elif self.codebook.device.type == "cuda":
        sub_recon = self._decode_cuda(sub_codebook, sub_code)
      recon[:, mask] = sub_recon
    return recon

  def precompute_adc3(self, x, return_labels=False):
    d_vector, n_data = x.shape
    assert d_vector == self.d_vector
    
    labels = self.codebook_selector.predict(x)    
    unique_labels, counts = labels.unique(return_counts=True)
    n_unique = unique_labels.shape[0]

    precomputed = torch.zeros(
      self.n_subvectors,
      n_data,
      self.n_clusters,
      device=self.codebook.device
    )
    
    mask = labels[:, None] == unique_labels[None]
    xs = [ x[:, mask[:, i]].T for i in range(n_unique)]
    lens = [i.shape[0] for i in xs]

    padded_x = (
      pad_sequence(xs, batch_first=True)
      .transpose(-1, -2)
      .reshape(n_unique * self.n_subvectors, self.d_subvector, -1)
    )

    codebook = (
      self.codebook[unique_labels]
      .reshape(n_unique * self.n_subvectors, self.d_subvector, self.n_clusters)
    )

    pcd = self.kmeans.sim(padded_x, codebook, normalize=False)
    pcd = pcd.reshape(n_unique, self.n_subvectors, -1, self.n_clusters)
    for i, label in enumerate(unique_labels):
      sub_mask = mask[:, i]
      precomputed[:, sub_mask] = pcd[i, :, :lens[i] ]
    if return_labels:
      return precomputed, labels
    else:
      return precomputed

  def precompute_adc2(self, x, return_labels=False):
    d_vector, n_data = x.shape
    assert d_vector == self.d_vector
    
    labels = self.codebook_selector.predict(x)
    unique_labels, counts = labels.unique(return_counts=True)
    
    precomputed = torch.zeros(
      self.n_subvectors,
      n_data,
      self.n_clusters,
      device=self.codebook.device
    )
    
    mask = labels[:, None] == unique_labels[None]
    for i, label in enumerate(unique_labels):
      
      sub_mask = mask[:, i]
      sub_x = x[:, sub_mask]
      sub_x = sub_x.reshape(self.n_subvectors, self.d_subvector, -1)
      sub_codebook = self.codebook[label]
      sub_precomputed = self.kmeans.sim(sub_x, sub_codebook, normalize=False)
      precomputed[:, sub_mask] = sub_precomputed
    if return_labels:
      return precomputed, labels
    else:
      return precomputed
      
  def precompute_adc(self, x, return_labels=False):
    """
      x: shape : [d_vector, n_data]
    """
    d_vector, n_data = x.shape
    assert d_vector == self.d_vector

    labels = self.codebook_selector.predict(x) #[n_data]
    unique_labels, counts = labels.unique(return_counts=True)
    groups = counts // self.group_size
    unique_groups = groups.unique()
    precomputed = torch.zeros(
      self.n_subvectors,
      n_data,
      self.n_clusters,
      device=self.codebook.device
    )
    for group_index in unique_groups:
      group_unique_labels = unique_labels[groups == group_index]
      n_gul = group_unique_labels.shape[0]

      mask = labels[:, None] == group_unique_labels[None, :] #[n_data, n_gul]
      mask2 = mask.sum(dim=1).bool() #[n_data]
      sub_x = x[:, mask2]
      sub_labels = labels[mask2]

      sub_codebook = self.codebook[group_unique_labels] #[n_gul, n_subvectors, d_subvector, n_clusters]
      sub_codebook = sub_codebook.reshape(-1, self.d_subvector, self.n_clusters)# [n_gul*n_subvectors, d_subvector, n_clusters]
      padded_x = [sub_x[:, sub_labels == lab].T for lab in group_unique_labels]
      del sub_x, sub_labels

      len_x = [padded_x[i].shape[0] for i in range(n_gul)]
      padded_x = (
        pad_sequence(padded_x, batch_first=True) #[n_gul, max_n_sub_x, d_vector]
        .transpose(-1, -2) #[n_gul, d_vector, max_n_sub_x]
        .reshape(n_gul * self.n_subvectors, self.d_subvector, -1) 
      ) #[n_gul* n_subvectors, d_subvector, max_n_sub_x]

      sub_precomputed = self.kmeans.sim(padded_x, sub_codebook, normalize=False) #[n_gul*n_subvectors, max_n_sub_x, n_clusters]
      del sub_codebook, padded_x
      sub_precomputed = sub_precomputed.reshape(n_gul, self.n_subvectors, -1, self.n_clusters) #[n_gul,n_subvectors, max_n_sub_x, n_clusters]
      for i in range(n_gul):
        lab = group_unique_labels[i]
        subsub_precomputed = sub_precomputed[i][:, :len_x[i]] #[n_subvectors, n_subsub_x, n_clusters]
        sub_mask = mask[:, i]
        precomputed[:, sub_mask] = subsub_precomputed
      del sub_precomputed
    if return_labels:
      return precomputed, labels
    else:
      return precomputed