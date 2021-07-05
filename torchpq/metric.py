import torch

def cosine_similarity(a, b, normalize=True, inplace=False):
  """
    Compute batched cosine similarity between 'a' and 'b'
    a: torch.Tensor, shape : [l, d_vector, m]
    b: torch.Tensor, shape : [l, d_vector, n]
    normalize: bool, default : True
      if True, a and b will be normalized to norm = 1
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

def negative_squared_l2_distance(a, b, inplace=False, use_tensor_core=False):
    """
      Compute batched negative squared euclidean (l2) distance between 'a' and 'b'
      a: torch.Tensor, shape : [l, d_vector, m]
      b: torch.Tensor, shape : [l, d_vector, n]
      inplace: bool, default : False
      returns: torch.Tensor, shape : [l, m, n]
    """
    # peak mem usage: m*n*4 + max(m,n)*4 + inplace ? 0: (m+n)*d*4
    if use_tensor_core:
      y = a.transpose(-2, -1).half() @ b.half()
      y = y.float()
      print(y)
    else:
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