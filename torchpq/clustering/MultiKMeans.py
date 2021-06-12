import torch
import torch.nn as nn
import numpy as np
import math
from time import time

from .kernels import MaxSimCUDA
from .kernels import ComputeCentroidsCUDA
from ..CustomModule import CustomModule

class MultiKMeans(CustomModule):
  """
  Run multiple independent K-means algorithms in parallel.
  Parameters:
    n_clusters: int, 
      Number of clusters

    max_iter: int, default: 100
      Maximum number of iterations

    tol: float, default: 0.0001
      Tolerance

    n_redo: int, default: 1
      Number of time k-means will be run with differently intialized centroids.
      the centroids with the lowest inertia will be selected as a final result.

    init_mode: {'random', 'kmeans++'}, default: 'random'
      Initialization method
      'random': randomly chose initial centroids from input data
      'kmeans++': use k-means++ algorithm to initialize centroids, slow when n_cluster is large, but converges faster)
    
    verbose: int, default: 0
      Verbosity

    distance: {'euclidean', 'cosine', 'manhattan'}, default: 'euclidean'
      Type of distance metric
      note: manhattan or L1 distance is only supported on GPU
      
  Attributes:
    centroids: torch.Tensor, shape: [d_vector, n_clusters]
      cluster centroids
  """
  def __init__(
    self,
    n_clusters,
    n_redo=1,
    max_iter=100,
    tol=1e-4,
    distance="euclidean",
    init_mode="random",
    verbose=0,
    sm_size=48*256*4,
   ):
    super(MultiKMeans, self).__init__()
    self.n_redo = n_redo
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.tol = tol
    self.verbose = verbose
    self.distance = distance
    self.init_mode = init_mode
    self.sm_size = sm_size
    self.arange = None
    if n_clusters < 4096:
      dk = n_clusters
    else:
      dk = 4096
    de = 1

    self.register_buffer("centroids", None)
    
    if torch.cuda.is_available():
      self.compute_centroids_cuda = ComputeCentroidsCUDA(
        de=de,
        dk=dk,
        sm_size=sm_size,
      )

      if distance in ["euclidean", "manhattan"]:
        distance = distance
      elif distance in ["cosine"]:
        distance = "inner"
      self.max_sim_cuda = MaxSimCUDA(
        distance=distance,
      )

  @staticmethod
  def remaining_memory(device):
    """
      Get remaining memory of GPU in bytes
    """
    # torch.cuda.synchronize()
    if device.type == "cpu":
      remaining = 32 * 1024 ** 3 # just a random large number
    elif device.type == "cuda":
      torch.cuda.empty_cache()
      total_memory = torch.cuda.get_device_properties(0).total_memory
      remaining = total_memory - torch.cuda.memory_reserved()
      # remaining = total_memory - torch.cuda.memory_allocated()
    return remaining

  @staticmethod
  def does_it_fit(size, device="cpu", dtype=torch.float):
    try:
      torch.empty(size, device=device, dtype=dtype)
    except:
      return False
    else:
      return True

  @staticmethod
  def calculate_error(a, b):
    """
      Compute L2 error between 'a' and 'b'
    """
    diff = a - b
    diff.pow_(2)
    return diff.sum()

  @staticmethod
  def calculate_inertia(a):
    return (-a).mean()

  @staticmethod
  def cos_sim(a, b, normalize=True, inplace=False):
    """
      Compute batched cosine similarity between 'a' and 'b'

      a: torch.Tensor, shape : [n_kmeans, d_vector, m]
      b: torch.Tensor, shape : [n_kmeans, d_vector, n]
      normalize: bool, default : True
        if True, a and b will be normalized to norm=1
      inplace: bool, default : False
      returns: torch.Tensor, shape : [l, m, n]
    """
    if normalize:
      a_norm = a.norm(dim=-2, keepdim=True) + 1e-8 #[l, m] <l*(m*4)>
      b_norm = b.norm(dim=-2, keepdim=True) + 1e-8 #[l, n] <l*(m*4 + n*4)>
      if inplace:
        # memory consump: m + n + (m * n)
        a.div_(a_norm)
        b.div_(b_norm)
      else:
        # memory consum: m + n + (m * n) + m*d + n*d
        a = a / a_norm #[l, d_vector, m], l*(<m*4 + n*4> + <m*d*4>)
        b = b / b_norm #[l, d_vector, n], l*(<m*4 + n*4> + <(m+n)*d*4>)
    prod = a.transpose(-2, -1) @ b #[l, m, n], <m*n*4 + m*4 + n*4> + <(m+n)*d*4>
    if inplace and normalize:
      a.mul_(a_norm)
      b.mul_(b_norm)
    return prod

  @staticmethod
  def euc_sim(a, b, inplace=False):
    """
      Compute batched negative squared euclidean distance between 'a' and 'b'
      a: torch.Tensor, shape : [l, d_vector, m]
      b: torch.Tensor, shape : [l, d_vector, n]
      inplace: bool, default : False
      returns: torch.Tensor, shape : [l, m, n]
    """
    # peak mem uwage: m*n*4 + max(m,n)*4 + inplace ? 0: (m+n)*d*4
    y = a.transpose(-2, -1) @ b # [m, n] <m*n*4>
    y.mul_(2)
    if inplace:
      a.pow_(2)
      b.pow_(2)
    else:
      a = a ** 2 #[m, d], <m*n*4 + m*d*4>
      b = b ** 2 #[n, d], <m*n*4 + n*d*4 + m*d*4>
    a2 = a.sum(dim=-2)[..., :, None] #? [m], <m*n*4 + m*4> + <n*d*4 + m*d*4>
    y.sub_(a2)
    del a2
    b2 = b.sum(dim=-2)[..., None, :] #[n], <m*n*4 + n*4> + <n*d*4 + m*d*4>
    y.sub_(b2)
    if inplace:
      a.sqrt_()
      b.sqrt_()
    return y
  
  def sim(self, a, b, inplace=False, normalize=True):
    """
      Compute batched similarity between 'a' and 'b', the type of distance metric is specified in __init__ method
      a: torch.Tensor, shape : [l, d, m]
      b: torch.Tensor, shape : [l, d, n]
      returns: torch.Tensor, shape : [l, m, n]
    """
    if self.distance == "euclidean":
      return self.euc_sim(a, b, inplace=inplace)
    elif self.distance == "cosine":
      return self.cos_sim(a, b, inplace=inplace, normalize=normalize)
    elif self.distance == "inner":
      return self.cos_sim(a, b, inplace=inplace, normalize=False)

  ### Need more testing:
  def kmeanspp(self, data):
    """
      Initialize centroids with k-means++ algorithm
      data: torch.Tensor, shape : [l, d_vector, n_data]
      returns: torch.Tensor, shape : [l, d_vector, n_clusters]
    """
    l, d_vector, n_data = data.shape
    if self.distance == "cosine":
      data_norm = data.norm(dim=-2, keepdim=True) + 1e-8
      data.div_(data_norm)
    centroids = torch.zeros(l, d_vector, self.n_clusters, device=data.device, dtype=data.dtype)
    #Select initial centroid
    centroids[:, :, 0] = data[:, :, np.random.randint(n_data)]
    for i in range(1, self.n_clusters):
      current_centroids = centroids[:, :, :i].contiguous()
      if data.device.type == "cpu":
        sims = self.sim(data, current_centroids ) #[l,m,n]
        max_sims_v, max_sims_i = sims.max(dim=-1) #[l,m]
      elif data.device.type == "cuda":
        max_sims_v, max_sims_i = self.max_sim_cuda(data, current_centroids, dim=2, mode="tn")
      index = max_sims_v.argmin(dim=-1) #[l]
      arange = torch.arange(l, device=device)
      new_centroid = data[arange, :, index] #[l, d_vector]
      centroids[:, :, i] = new_centroid
    if self.distance == "cosine":
      data.mul_(data_norm)
    return centroids

  def initialize_centroids(self, data):
    """
      Initialize centroids with init_method specified in __init__
      data: torch.Tensor, shape : [l, d_vector, n_data]
      return: torch.Tensor, shape: [l, d_vector, n_clusters]
    """
    l, d_vector, n_data = data.shape
    if self.init_mode == "random":
      random_index = np.random.choice(
        n_data,
        size=[self.n_clusters],
        replace=False
      )
      centroids = data[:, :, random_index]
      if self.verbose >= 1:
        print("centroids are randomly initialized.")

    elif self.init_mode == "kmeans++":
      centroids = self.kmeanspp(data)
      if self.verbose >= 1:
        print("kmeans++ initialization is done!")
    return centroids

  def get_labels(self, data, centroids):
    """
      data: torch.Tensor, shape : [l, d_vector, n_data]
      centroids: torch.Tensor, shape : [l, d_vector, n_clusters]
      return: torch.Tensor, shape : [l, n_data], dtype: long
    """
    #memory requirement: 

    l, d, m = data.shape
    l, d, n = centroids.shape

    remaining = self.remaining_memory(data.device)# - 1024*3

    if self.distance == "euclidean":
      required = l*(m*n + max(m, n) + m*d + n*d) * data.element_size()
    elif self.distance in ["cosine", "inner"]:
      required = l*((m*n) + (m+n)*(d+1)) * data.element_size()
    if remaining >= required:
      sims = self.sim(data, centroids, inplace=False) #[l, m, n]
      maxsims, labels = sims.max(dim=-1) #[l, m]
      return (maxsims, labels)
    else:
      if data.device.type == "cuda":
        if self.distance == "cosine":
          d_norm = data.norm(dim=-2, keepdim=True) + 1e-8
          c_norm = centroids.norm(dim=-2, keepdim=True) + 1e-8
          data.div_(d_norm)
          centroids.div_(c_norm)
        maxsims, labels = self.max_sim_cuda(data, centroids, dim=2, mode="tn")
        if self.distance == "cosine":
          data.mul_(d_norm)
          centroids.mul_(c_norm)
      elif data.device.type == "cpu":
        ## doing in seperate chunks
        n_partitions = 1
        for i in range(16):
          sub_m = math.ceil(m / n_partitions)
          if self.distance == "euclidean":
            required = l*(sub_m*n + max(sub_m, n)) * data.element_size() + m*8 # +sub_m*d*4
          elif self.distance in ["cosine", "inner"]:
            required = l*(sub_m*n + sub_m+n) * data.element_size() + m*8# +sub_m*d*4
          # print("required, remaining, n_p", required / 1024**3, remaining / 1024**3, n_partitions)
          if self.does_it_fit(required // 4, device=data.device, dtype=torch.float):
            break
          n_partitions *= 2

        maxsims = torch.zeros(l, m, device=data.device, dtype=torch.float)
        labels = torch.zeros(l, m, device=data.device, dtype=torch.long)
        for i in range(n_partitions):
          start = i*sub_m
          if i == n_partitions - 1:
            end = m - 1
          else:
            end = (i+1)*sub_m
          sub_data = torch.narrow(data, dim=-1, start=start, length=end-start) #[l, d, sub_m]
          # sub_data = data[:, start:end] #[l, d, sub_m]
          sub_sims = self.sim(sub_data, centroids, inplace=True) #[l, sub_m, n]
          del sub_data
          sub_maxsims, sub_labels = sub_sims.max(dim=-1) #[l, sub_m]
          del sub_sims
          labels[:, start:end] = sub_labels
          maxsims[:, start:end] = sub_maxsims
          del sub_labels
      return (maxsims, labels)

  def compute_centroids_loop(self, data, labels):
    """
      data: torch.Tensor, shape : [l, d_vector, n_data]
      labels: torch.Tensor, shape : [l, n_data]
      return: torch.Tensor, shape : [l, d_vector, n_clusters]
    """
    ### Naive method with loop
    l, d, m = data.shape
    centroids = torch.zeros(l, d, self.n_clusters, device=data.device, dtype=data.dtype)
    for j in range(l):
      unique_labels, counts = labels[j].unique(return_counts=True)
      for i, count in zip(unique_labels, counts):
        centroids[j, :, i] = data[j, :, labels[j]==i].sum(dim=1) / count
    return centroids

  def compute_centroids(self, data, labels):
    """
      data: torch.Tensor, shape : [l, d_vector, n_data]
      labels: torch.Tensor, shape : [l, n_data]
      return: torch.Tensor, shape: [l, d_vector, n_clusters]
    """
    if data.device == torch.device("cpu"):
      centroids = self.compute_centroids_loop(data, labels)
    else:
      centroids = self.compute_centroids_cuda(data, labels, k=self.n_clusters)
    return centroids

  def _compute_centroids_hungry(self, data, labels):
    ### Memory hungry method
    # expanded_labels = labels[None].expand(self.n_clusters, -1) #[k, n], k=n_clusters <>
    if self.arange is None\
    or self.arange.dtype != data.dtype\
    or self.arange.device != data.device:
      self.arange = torch.arange(self.n_clusters, device=data.device) #[k] <k*8>
    
    mask = labels[None, :] == self.arange[:, None] #[k, n] <k*n*1 + k*8>
    mask_sum = mask.sum(dim=-1) #[k] <k*n*1 + k*12>
    mask = mask.float() # <k*n*5 + k*12> LARGEST MEMORY USE!!!
    centroids = mask @ data # <k*n*4 + k*12 + k*d*4>
    del mask

    centroids.div_(mask_sum[..., :, None]) # <k*d*4 + k*12>
    del mask_sum
    nan_mask = centroids!=centroids #[k, d] # <k*d*8>
    centroids[nan_mask] = 0 # remove NaNs

    return centroids

  def fit(self, data, centroids=None):
    """
      Perform K-means clustering, and return final labels
      data: torch.Tensor, shape : [l, d_vector, n_data]
      return: torch.Tensor, shape : [l, n_data], dtype: long
    """
    assert data.is_contiguous(), "use .contiguous()"

    best_centroids = None
    best_error = 1e32
    best_labels = None
    best_inertia = 1e32
    tm = time()
    for i in range(self.n_redo):
      tm_i = time()
      if centroids is None:
        centroids = self.initialize_centroids(data)
      for j in range(self.max_iter):
        # 1 iteration of clustering
        maxsims, labels = self.get_labels(data, centroids) #top1 search
        new_centroids = self.compute_centroids(data, labels)
        error = self.calculate_error(centroids, new_centroids)
        centroids = new_centroids
        inertia = self.calculate_inertia(maxsims)
        if self.verbose >= 3:
          print(f"----iteration {j} of {i}th redo, error={error.item()}, inertia={inertia.item()}")
        if error <= self.tol:
          break

      if inertia < best_inertia:
        best_centroids = centroids
        best_error = error
        best_labels = labels
        best_inertia = inertia
      if self.verbose >= 2:
        print(f"--{i}th redo finished, error: {error.item()}, inertia: {inertia.item()}time spent:{round(time()-tm_i, 4)} sec")

    self.register_buffer("centroids", best_centroids)
    if self.verbose >= 1:
      print(f"finished {self.n_redo} redos in {round(time()-tm, 4)} sec, final_inertia: {best_inertia}")
    return best_labels

  def predict(self, query):
    """
      Predict closest cluster center each sample in query belongs to.
      query: torch.Tensor, shape : [l, d_vector, n_query]
      return: torch.Tensor, shape : [l, n_query]
    """
    assert self.centroids is not None, "kmeans is not trained"
    _, labels = self.get_labels(query, self.centroids)
    return labels

  def topk(self, query, k):
    """
      Predict the top-k closest cluster centers of each sample in query
      query: torch.Tensor, shape : [l, d_vector, n_query]
      k: int, should be in range [1, n_centroids]
    """
    assert self.centroids is not None, "kmeans is not trained"
    assert k <= self.n_centroids, "k is too large"
    sims = self.sim(query, self.centroids) #[l, n_query, n_clusters]
    topkv, topki = sims.topk(dim=-1, k=k) #[l, n_query, k]
    return (topkv, topki)