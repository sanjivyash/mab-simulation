import numpy as np
from algorithms import Algorithms

class ScaleOptim(Algorithms):
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
      self.arms      = len(lines)
      self.support   = np.array([0, 1])
      self.dist      = np.array([[1-float(l), float(l)] for l in lines])
      self.cumdist   = np.cumsum(self.dist, axis=1)

    # initialize the methods inherited from Algorithms class
    super().__init__()

  def __call__(self):
    np.random.seed(self.rs)
    self.ucb()
