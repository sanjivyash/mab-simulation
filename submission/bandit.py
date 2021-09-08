import argparse

from task1 import BerSampling
from task2 import ScaleOptim
from task3 import GenSampling
from task4 import HighOptim

algo = ["epsilon-greedy-t1", "ucb-t1", "kl-ucb-t1", "thompson-sampling-t1", "ucb-t2", "alg-t3", "alg-t4"]
parser = argparse.ArgumentParser(description="bandit instance arguments")

parser.add_argument(
  "--instance", type=str, 
  metavar="in", default=None, 
  help="path to the instance file"
)

parser.add_argument(
  "--algorithm", type=str, 
  metavar="al", default=None, 
  help=" | ".join(algo)
)

parser.add_argument(
  "--randomSeed", type=int, 
  metavar="rs", default=0, 
  help="seed for random functions"
)

parser.add_argument(
  "--epsilon", type=float,
  metavar="ep", default=0.02,
  help="parameter in [0,1] for eps-greedy algos"
)

parser.add_argument(
  "--scale", type=float,
  metavar="c", default=2,
  help="scale for Task 2"
)

parser.add_argument(
  "--threshold", type=float,
  metavar="th", default=0,
  help="parameter in [0,1] for Task 4"
)

parser.add_argument(
  "--horizon", type=int,
  metavar="hz", default=0,
  help="horizon for different algos"
)

args = parser.parse_args()
task = args.algorithm.split("-")[-1]

assert args.instance, "please provide a file path"
assert args.randomSeed >= 0, "please provide a non-negative random seed"
assert args.horizon > 0, "please provide a positive bandit horizon"

if task == "t1":
  BerSampling(
    args.instance,
    args.algorithm,
    args.randomSeed,
    args.epsilon,
    args.scale,
    args.threshold,
    args.horizon
  )()

if task == "t2":
  ScaleOptim(
    args.instance,
    args.algorithm,
    args.randomSeed,
    args.epsilon,
    args.scale,
    args.threshold,
    args.horizon
  )()

if task == "t3":
  GenSampling(
    args.instance,
    args.algorithm,
    args.randomSeed,
    args.epsilon,
    args.scale,
    args.threshold,
    args.horizon
  )()

if task == "t4":
  HighOptim(
    args.instance,
    args.algorithm,
    args.randomSeed,
    args.epsilon,
    args.scale,
    args.threshold,
    args.horizon
  )()