import os
import numpy as np

class Algorithms:
  '''
    List of attributes needed
    - ins             file instance
    - al              algorithm to be used
    - rs              random seed
    - ep              epsilon
    - c               scale
    - th              threshold
    - hz              horizon
    - arms            number of arms
    - support         support of distribution
    - dist            distribution of the arms
    - cumdist         cumulative distribution
  '''

  def _sample(self, i):
    assert 0 <= i < self.arms, "arm number out of range"

    p              = np.random.random()
    rew            = self.support[np.searchsorted(self.cumdist[i], p)]
    self.rew      += rew
    total          = self.avg[i] * self.count[i] + rew
    self.count[i] += 1
    self.avg[i]    = total / self.count[i]

  def _search(self, i, t):
    assert 0 <= i < self.arms, "arm number out of range"

    dx = 1e-7
    f  = lambda x: min(max(x, dx), 1 - dx)
    kl = lambda p, q: f(p) * np.log(f(p) / f(q)) + f(1-p) * np.log(f(1-p) / f(1-q))

    avg, count = self.avg[i], self.count[i]
    start, end = avg, 1
    bound      = np.log(t) + self.c * np.log(np.log(t))

    while start <= end - dx:
      mid     = (start + end)/2
      if kl(avg, mid) > bound / count:
        end   = mid
      else:
        start = mid

    return (start + end)/2

  def _output(self):
    ins     = os.path.relpath(self.ins, os.path.dirname(__file__))
    
    if self.al == "alg-t4":
      reg   = np.max(np.sum((self.support > self.th) * self.dist, axis=1)) * self.hz - self.rew
      reg   = np.round(reg, 4)
      print(f"{ins}, {self.al}, {self.rs}, {self.ep}, {self.c}, {self.th}, {self.hz}, {reg}, {self.rew}")
    
    else:
      reg   = np.max(np.sum(self.support * self.dist, axis=1)) * self.hz - self.rew
      reg   = np.round(reg, 4)
      print(f"{ins}, {self.al}, {self.rs}, {self.ep}, {self.c}, {self.th}, {self.hz}, {reg}, 0")

  def eps(self):
    self.rew       = 0
    self.avg       = np.zeros(self.arms)
    self.count     = np.zeros(self.arms)
    
    # sample each arm once at the start 
    for i in range(self.arms):
      self._sample(i)

    for i in range(self.arms, self.hz):
      ep     = np.random.random()

      if ep  < self.ep:   # explore
        r    = np.random.random()
        arm  = int(r * self.arms)
      else:               # exploit
        arm  = np.argmax(self.avg)

      self._sample(arm)
    self._output()

  def ucb(self):
    self.rew       = 0
    self.avg       = np.zeros(self.arms)
    self.count     = np.zeros(self.arms)

    # sample each arm once at the start 
    for i in range(self.arms):
      self._sample(i)

    for i in range(self.arms, self.hz):
      t   = i + 1    # round number
      ucb = self.avg + np.sqrt(self.c * np.log(t) / self.count)
      arm = np.argmax(ucb)
      self._sample(arm)

    self._output()

  def kl_ucb(self):
    self.rew       = 0
    self.avg       = np.zeros(self.arms)
    self.count     = np.zeros(self.arms)    

    # sample each arm once at the start 
    for i in range(self.arms):
      self._sample(i)

    for i in range(self.arms, self.hz):
      t      = i + 1    # round number
      kl_ucb = np.array([self._search(i, t) for i in range(self.arms)])
      arm    = np.argmax(kl_ucb)
      self._sample(arm)

    self._output()

  def thompson(self):
    self.rew     = 0
    self.success = np.ones(self.arms)
    self.failure = np.ones(self.arms)

    for i in range(self.hz):
      arm, val  = -1, -1

      for j in range(self.arms):
        a, b    = self.success[j], self.failure[j]
        sample  = np.random.beta(a, b)   # sample each arm
        if val  < sample:
          val   = sample
          arm   = j                      # choose best arm

      p         = np.random.random()
      rew       = self.support[np.searchsorted(self.cumdist[arm], p)]

      if rew > self.th:
        self.rew          += 1
        self.success[arm] += 1
      else:
        self.failure[arm] += 1

    self._output()
