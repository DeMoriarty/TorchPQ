from . import kmeans
from . import kernels

from .IVFPQBase import IVFPQBase
from .PQ import PQ

from .IVFPQ import IVFPQ
from .IVFPQR import IVFPQR

try:
  import cupy
except ModuleNotFoundError:
  raise ModuleNotFoundError("cupy is not installed, please visit https://pypi.org/project/cupy/")