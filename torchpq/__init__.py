from .IVFPQBase import IVFPQBase
from .IVFPQ import IVFPQ
from .IVFPQR import IVFPQR
from .PQ import PQ
from . import kmeans
from . import kernels

try:
  import cupy
except ModuleNotFoundError:
  raise ModuleNotFoundError("cupy is not installed, please visit https://pypi.org/project/cupy/")