import numpy as np
from algorithms import Algorithms

class BerSampling(Algorithms):
  def __init__(self, ins, al, rs, ep, c, th, hz):
    self.ins = ins  # file path
    self.al  = al   # algorithm
    self.rs  = rs   # random seed
    self.ep  = ep   # epsilon value
    self.c   = c    # scale (only for output)
    self.th  = th   # threshold (only for Thompson Sampling)
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
    funcmap = {
      "epsilon-greedy-t1": self.eps,
      "ucb-t1": self.ucb,
      "kl-ucb-t1": self.kl_ucb,
      "thompson-sampling-t1": self.thompson
    }

    np.random.seed(self.rs)
    assert self.al in funcmap, "incorrect algorithm specified in task 1"
    funcmap[self.al]()
