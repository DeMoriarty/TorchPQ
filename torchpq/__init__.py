from torchpq.IVFPQBase import IVFPQBase
from torchpq.IVFPQ import IVFPQ
from torchpq.IVFPQR import IVFPQR
from torchpq.PQ import PQ
try:
  import cupy
except ModuleNotFoundError:
  raise ModuleNotFoundError("cupy is not installed, please visit https://pypi.org/project/cupy/")