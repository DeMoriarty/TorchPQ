import torch
from ..kernels import TopkSelectCuda, Top32SelectCuda, Top1SelectCuda

class Topk:
  def __init__(self):

    self._top1_cuda = Top1SelectCuda(
      tpb = 256,
      queue_capacity = 1,
      buffer_size = 4,
    )
    self._top32_cuda = Top32SelectCuda(
      tpb = 256,
      queue_capacity = 1,
      buffer_size = 4,
    )
    self._top64_cuda = TopkSelectCuda(
      tpb = 64,
      queue_capacity = 4,
      buffer_size = 4,
    )
    self._top128_cuda = TopkSelectCuda(
      tpb = 128,
      queue_capacity = 4,
      buffer_size = 4,
    )
    self._top256_cuda = TopkSelectCuda(
      tpb = 256,
      queue_capacity = 4,
      buffer_size = 4,
    )
    self._top512_cuda = TopkSelectCuda(
      tpb = 512,
      queue_capacity = 2,
      buffer_size = 2,
    )
    self._top1024_cuda = TopkSelectCuda(
      tpb = 1024,
      queue_capacity = 2,
      buffer_size = 2,
    )

  def __call__(self, x, k=1, dim=1):
    if dim == -1:
      dim = 1
    assert dim == 1, "only support last dimention"
    assert len(x.shape) == 2, "only support 2d tensors"
    assert x.is_contiguous(), "x is not contiguous"
    assert k >= 1
    assert x.device.type == "cuda"
    if k == 1:
      # return torch.max(x, dim=dim, keepdim=True)
      return self._top1_cuda(x, dim=dim, k=k)
    elif k <= 32:
      return self._top32_cuda(x, dim=dim, k=k)
    elif k <= 64:
      return self._top64_cuda(x, dim=dim, k=k)
    elif k <= 128:
      return self._top128_cuda(x, dim=dim, k=k)
    elif k <= 256:
      return self._top256_cuda(x, dim=dim, k=k)
    elif k <= 512:
      return self._top512_cuda(x, dim=dim, k=k)
    elif k <= 1024:
      return self._top1024_cuda(x, dim=dim, k=k)
    else:
      return torch.topk(x, dim=dim, k=k)
