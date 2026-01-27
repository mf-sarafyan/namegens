import torch
import torch.nn.functional as F

g = torch.Generator().manual_seed(2147483647) 

class Linear:
  
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
    self.bias = torch.zeros(fan_out) if bias else None
  
  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out
  
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  def __call__(self, x):
    # calculate the forward pass
    if self.training:
      xmean = x.mean(0, keepdim=True) # batch mean
      xvar = x.var(0, keepdim=True) # batch variance
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    # update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]


class LayerNorm1d:
  
  def __init__(self, dim, eps=1e-5):
    self.dim = dim
    self.eps = eps
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
  
  def __call__(self, x):
    # LayerNorm normalizes across the last dimension (features)
    # Works for both 2D (batch, features) and 3D (batch, seq_len, features) inputs
    # Calculate mean and variance across the last dimension
    xmean = x.mean(-1, keepdim=True)  # mean across last dimension
    xvar = x.var(-1, keepdim=True, unbiased=False)  # variance across last dimension
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]

class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []

class Dropout:
  def __init__(self, p=0.5):
    self.p = p
    self.training = True
  
  def __call__(self, x):
    if self.training:
      # Create dropout mask: (1-p) probability of keeping
      mask = (torch.rand_like(x) > self.p).float()
      # Scale by 1/(1-p) to maintain expected value
      self.out = x * mask / (1 - self.p)
    else:
      self.out = x
    return self.out
  
  def parameters(self):
    return []