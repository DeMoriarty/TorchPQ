class SQ(nn.Module):
  """
  scalar qantizer
  Parameters:
    bits: int, default : 8
      desired bit size of scalars, can be one of 4, 8, 16, 32
    
    mode: str, default : minmax
      mode of uniform scalar quantization, supported modes are "minmax" and "meanstd"

    alpha: float, default : 1.0
      used in "meanstd"
    
  """
  def __init__(
    self,
    bits=8,
    alpha=1.0,
    mode="minmax",
    ):
    super(SQ, self).__init__()
    self.bits = bits
    self.mode = mode
    self.alpha = alpha
    self.n_bins = 2 ** bits
    
    self.register_buffer("lower", None)
    self.register_buffer("upper", None)
    self.register_buffer("binsize", None)
    self.register_buffer("range", None)
    self.register_buffer("is_trained", torch.tensor(False))


  def _train_minmax(self, input):
    mins = input.min(dim=-1)[0]
    maxs = input.max(dim=-1)[0]
    self.register_buffer("lower", mins )
    self.register_buffer("upper", maxs )
    self.register_buffer("range", (self.upper - self.lower + 1e-8))

  def _train_meanstd(self, input):
    means = input.mean(dim=-1)
    stds = input.std(dim=-1)
    self.register_buffer("lower", means - self.alpha * stds )
    self.register_buffer("upper", means + self.alpha * stds )
    self.register_buffer("range", (self.upper - self.lower + 1e-8))
  
  def train(self, input):
    """
      Train scalar quantizer with input data statistics
      input: torch.Tensor, shape : [d_vector, n_data], dtype : float32
    """
    if self.bits == 4:
      assert input.shape[0] % 2 == 0, "d_vector needs to be divisible by 2"

    if self.bits <= 8:
      if self.mode == "minmax":
        self._train_minmax(input)

      elif self.mode == "meanstd":
        self._train_meanstd(input)
      
    self.is_trained.data = torch.tensor(True)

  def _encode_32bit(self, input):
    return input

  def _decode_32bit(self, code):
    return code
  
  def _encode_16bit(self, input):
    return input.half()

  def _decode_16bit(self, code):
    return code.float()

  def _encode_8bit(self, input):
    input = input - self.lower[:, None]
    input.div_(self.range[:, None])
    input.mul_(self.n_bins)
    input.round_()
    input.clamp_(min=0, max=self.n_bins - 1)
    code = input.byte()
    return code

  def _decode_8bit(self, code):
    output = code.float()
    output.div_(self.n_bins)
    output.mul_(self.range[:, None])
    output.add_(self.lower[:, None])
    return output

  def _encode_4bit(self, input):
    d_vector = input.shape[0]
    input = input - self.lower[:, None]
    input.div_(self.range[:, None])
    input.mul_(self.n_bins)
    input.round_()
    input.clamp_(min=0, max=self.n_bins - 1)
    first = input[:d_vector // 2,]
    last  = input[d_vector // 2:,]
    code = first * self.n_bins + last
    code = code.byte()
    return code

  def _decode_4bit(self, code):
    code = code.float()
    first = code // self.n_bins
    last = code % self.n_bins
    output = torch.cat([first, last], dim=0)
    del first, last, code
    output.div_(self.n_bins)
    output.mul_(self.range[:, None])
    output.add_(self.lower[:, None])
    return output

  def encode(self, input):
    """
      input: torch.Tensor, shape : [d_vector, n_data], dtype : float32
    """
    assert self.is_trained == True, "SQ is not trained"
    if self.bits == 4:
      return self._encode_4bit(input)
    elif self.bits == 8:
      return self._encode_8bit(input)
    elif self.bits == 16:
      return self._encode_16bit(input)
    elif self.bits == 32:
      return self._encode_32bit(input)

  def decode(self, code):
    """
      code: torch.Tensor, shape : [code_size, n_data], dtype : float32 / float16 / uint8
    """
    assert self.is_trained == True, "SQ is not trained"
    if self.bits == 4:
      return self._decode_4bit(code)
    elif self.bits == 8:
      return self._decode_8bit(code)
    elif self.bits == 16:
      return self._decode_16bit(code)
    elif self.bits == 32:
      return self._decode_32bit(code)