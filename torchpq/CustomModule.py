import torch
import torch.nn as nn

class CustomModule(nn.Module):
  def __init__(self):
    super(CustomModule, self).__init__()

  def load_state_dict(self, state_dict):
    for k, v in state_dict.items():
      if "." not in k:
        assert hasattr(self, k), f"attribute {k} does not exist"
        delattr(self, k)
        self.register_buffer(k, v)
        
    for name, module in self.named_children():
      sd = {k.replace(name+".", "") : v for k, v in state_dict.items() if k.startswith(name+".")}
      module.load_state_dict(sd)
      # else:
      #   module = k.split('.')[0]
      #   assert hasattr(self, module), f"module {module} does not exist"
      #   module = getattr(self, module)
      #   k = ".".join(k.split('.')[1:])
      #   module.load_state_dict( {k: v} )