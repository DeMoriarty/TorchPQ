import torch
import torch.nn as nn

class CustomModule(nn.Module):
  def __init__(self):
    super(CustomModule, self).__init__()

  def print_message(self, text, min_verbosity=0):
    if hasattr(self, "verbose"):
      if self.verbose < min_verbosity:
        return
    print(f"{type(self).__name__}: {text}")

  def load_state_dict(self, state_dict):
    for k, v in state_dict.items():
      if "." not in k:
        assert hasattr(self, k), f"attribute {k} does not exist"
        delattr(self, k)
        self.register_buffer(k, v)
        
    for name, module in self.named_children():
      sd = {k.replace(name+".", "") : v for k, v in state_dict.items() if k.startswith(name+".")}
      module.load_state_dict(sd)