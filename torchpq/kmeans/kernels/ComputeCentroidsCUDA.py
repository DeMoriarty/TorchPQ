import torch
import cupy as cp
import numpy as np
import math
from .CustomKernel import CustomKernel
from torchpq.util import get_absolute_path

class ComputeCentroidsCUDA(CustomKernel):
  def __init__(
      self,
      de=16,
      dk=16,
      sm_size=48*256*4,
    ):
    super(ComputeCentroidsCUDA, self).__init__()
    self.de = de
    self.dk = dk
    assert dk * (de + 1) * 4 <= sm_size
    self.tpb = 256
    self.sm_size = sm_size

    with open(get_absolute_path("kmeans", "kernels", "ComputeCentroidsKernel.cu"), "r") as f:
      self.kernel = f.read()

    kernel = (self.kernel
      .replace("_DE_", str(de))
      .replace("_DK_", str(dk))
      .replace("_TPB_", str(self.tpb))
      .replace("_NITERS_", str(math.ceil(dk / self.tpb)))
    )

    self.fn = cp.RawKernel(
      kernel,
      'compute_centroids',
      # options=('--maxrregcount=255',),
      # backend='nvcc',
    )

    self.fn.max_dynamic_shared_size_bytes = sm_size
    # print(self.fn.attributes)

  def __call__(self, data, labels, k, centroids=None):
    """
      data: shape = [m, d_vector, n_data]   ###preferably [m, d, n_data]
      labels: shape = [m, n_data]
      centroids: shape = [m, n_clusters, d_vector]
    """
    m, d_vector, n_data = data.shape
    assert labels.shape[1] == n_data
    if centroids is not None:
      assert centroids.shape[0] == d_vector

      data = data.contiguous()
      labels = labels.contiguous()

    # data = data.transpose(1, 2).contiguous()

    centroids = torch.zeros(m, d_vector, k, device="cuda:0", dtype=torch.float32)
    threads_per_block = (self.tpb,)
    blocks_per_grid = (m, math.ceil(k / self.dk) , math.ceil(d_vector / self.de))
    # print("Blocks per grid", blocks_per_grid)

    torch.cuda.synchronize()
    self.fn(
      grid=blocks_per_grid,
      block=threads_per_block,
      shared_mem = self.dk * (self.de + 1) * 4,
      args=[
        data.data_ptr(),
        labels.data_ptr(),
        centroids.data_ptr(),
        m, # n_subquantizers / n_subvectors
        n_data,
        d_vector,
        k # n_clusters
        ],
      stream=self.stream
    )
    torch.cuda.synchronize()
    return centroids