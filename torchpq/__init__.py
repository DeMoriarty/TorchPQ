try:
  import cupy as cp
except ModuleNotFoundError:
  raise ModuleNotFoundError("cupy is not installed, please visit https://pypi.org/project/cupy/")
import torch
import torch.nn as nn
import numpy as np
import math
from time import time

from . import kmeans
from . import kernels

from .IVFPQBase import IVFPQBase
from .IVFPQTopk import IVFPQTopk
from .PQ import PQ

from .IVFPQ import IVFPQ
from .IVFPQR import IVFPQR
