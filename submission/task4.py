import numpy as np
from algorithms import Algorithms

class HighOptim(Algorithms):
  def __init__(self, ins, al, rs, ep, c, th, hz):
    self.ins = ins  # file path
    self.al  = al   # algorithm
    self.rs  = rs   # random seed
    self.ep  = ep   # epsilon value
    self.c   = c    # scale (only for output)
    self.th  = th   # threshold (only for output)
    self.hz  = hz   # horizon
    
    with open(self.ins) as f:
      lines          = f.readlines()
      self.arms      = len(lines) - 1
      self.support   = np.array([float(l) for l in lines[0].strip().split()])
      self.dist      = np.array([[float(l) for l in line.strip().split()] for line in lines[1:]])
      self.cumdist   = np.cumsum(self.dist, axis=1)
      self.avg       = np.zeros(self.arms)
      self.count     = np.zeros(self.arms)

    # initialize the methods inherited from Algorithms class
    super().__init__()

  def __call__(self):
    assert self.al == "alg-t4", "incorrect algorithm specified for task 4"
    np.random.seed(self.rs)
    self.thompson()