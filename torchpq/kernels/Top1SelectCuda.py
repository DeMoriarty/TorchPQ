import torch
import cupy as cp
import numpy as np
import math

from .CustomKernel import CustomKernel
from ..util import get_absolute_path

class Top1SelectCuda(CustomKernel):
  """
    tpb: threads per block, needs to be a power of 2 between 32 and 1024
    queue_capacity: capacity of thread queue
    buffer_size: number of elements each threads needs to prefetch

    What's new:
      optimize for k == 1
  """
  def __init__(self, tpb=256, queue_capacity=4, buffer_size=4):
    super().__init__()
    assert tpb >= 32
    assert self.next_power_of_2(tpb) == tpb
    assert queue_capacity >= 1
    assert buffer_size >= 1
    self.tpb = tpb
    self.queue_capacity = queue_capacity
    self.buffer_size = buffer_size
    self.n_warp = tpb // 32 #warps per block
    self.kernel_name = "top1_select"
    self.kernel_name_fp16 = "top1_select_fp16"

    with open(get_absolute_path("kernels", "cuda", f"{self.kernel_name}.cu"),'r') as f: ###
      self.kernel = f.read()
    
    self.kernel = (
      self.kernel
      .replace("_TPB_", str(tpb))
      .replace("_QCAP_", str(queue_capacity))
      .replace("_TN_", str(buffer_size))
    )

    self._fn_fp32 = cp.RawKernel(
      code=self.kernel,
      name=self.kernel_name,
      backend='nvrtc',
      options=(
        '--use_fast_math',
        '-lineinfo'
        # '--maxrregcount=128',
        #'-Xptxas',
        #'-dlcm=cg',
      )
    )
    self._fn_fp16 = cp.RawKernel(
      code=self.kernel,
      name=self.kernel_name_fp16,
      backend='nvrtc',
      options=(
        '--use_fast_math',
        '-lineinfo'
        # '--maxrregcount=128',
        #'-Xptxas',
        #'-dlcm=cg',
      )
    )
    # print(self._fn_fp32.attributes)

  @staticmethod
  def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))
  
  def __call__(self, x, k=128, dim=1):
    """
      x: shape = [m, n], dtype: float32
      k: 1 to 32
      dim: 1
    """
    assert len(x.shape) == 2
    assert x.dtype in [torch.float32, torch.float16]
    assert x.device.type == "cuda"
    # assert 1 <= k <= self.tpb
    assert k == 1
    assert dim == 1
    assert x.is_contiguous()
    k_pow_of_2 = self.next_power_of_2(k)

    m, n = x.shape
    threads_per_block = (self.tpb, )
    blocks_per_grid = (math.ceil(m / self.n_warp), )
    values = torch.empty(m, k_pow_of_2, device="cuda:0", dtype=x.dtype)
    values.fill_(float("-inf"))
    indices = torch.empty(m, k_pow_of_2, device="cuda:0", dtype=torch.long)


    if x.dtype is torch.float32:
      self._fn_fp32(
        grid = blocks_per_grid,
        block = threads_per_block,
        args = [
          x.data_ptr(),
          values.data_ptr(),
          indices.data_ptr(),
          m, n, k_pow_of_2
        ],
        stream=self.stream
      )
    elif x.dtype is torch.float16:
      self._fn_fp16(
        grid = blocks_per_grid,
        block = threads_per_block,
        args = [
          x.data_ptr(),
          values.data_ptr(),
          indices.data_ptr(),
          m, n, k_pow_of_2
        ],
        stream=self.stream
      )

    return values[:, :k], indices[:, :k]