import os
import shutil
import argparse
from pathlib import Path
from threading import Thread
import numpy as np

parser = argparse.ArgumentParser(description="run bandit.py")

parser.add_argument(
  "--task", type=int, 
  metavar="task", default=None, 
  help="task to run (1 to 4)"
)

parser.add_argument(
  "--plot", dest="plot",
   action="store_true",
   help="plot the results or not"
)

args = parser.parse_args()
assert args.task is not None, "please enter which task to run 1-4"
assert 1 <= args.task <= 4, "tasks between 1 and 4"

dirname   = os.path.dirname(__file__)
instances = os.path.join(dirname, "instances", f'instances-task{args.task}')
outfile   = os.path.join(dirname, f't{args.task}-out.txt')
Path(outfile).unlink(missing_ok=True)

##### TASK 1 ######
if args.task == 1:
  def sampler(seed, instance):
    num = instance.split("/")[-1].split(".")[0]
    tempfile = os.path.join(dirname, f'out-{num}-{seed}.txt')
    Path(tempfile).unlink(missing_ok=True)
    
    print(f'seed={seed} instance={instance}')
    algos = ["epsilon-greedy-t1", "ucb-t1", "kl-ucb-t1", "thompson-sampling-t1"]
    hzs   = [100, 400, 1600, 6400, 25600, 102400]

    for hz in hzs:
      for algo in algos:
        print(f'  hz={hz} algo={algo}')
        os.system(f"\
            python submission/bandit.py \
            --instance {instance} \
            --algorithm {algo} \
            --epsilon 0.02 \
            --randomSeed {seed} \
            --horizon {hz} >> {tempfile} \
          "
        )

  for file in os.listdir(instances):
    instance = os.path.join(instances, file)
    threads = []

    for seed in range(50):
      thread = Thread(target=sampler, args=(seed, instance), daemon=True)
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

    with open(outfile, 'a') as f:
      for seed in range(50):
        num = instance.split("/")[-1].split(".")[0]
        tempfile = os.path.join(f'out-{num}-{seed}.txt')

        with open(tempfile, 'r') as g:
          shutil.copyfileobj(g, f)

        Path(tempfile).unlink(missing_ok=True)
    print(f'{instance} resolved')


##### TASK 2 ######
if args.task == 2:
  def scaleopt(scale, instance):
    num = instance.split("/")[-1].split(".")[0]
    tempfile = os.path.join(dirname, f'out-{num}-{scale}.txt')
    Path(tempfile).unlink(missing_ok=True)

    for seed in range(50):
      os.system(f"\
          python submission/bandit.py \
          --instance {instance} \
          --algorithm ucb-t2 \
          --randomSeed {seed} \
          --scale {scale} \
          --horizon 10000 >> {tempfile} \
        "
      )

  for file in os.listdir(instances):
    instance = os.path.join(instances, file)
    c_opt, regret = 0, 10000
    threads = []

    for c in np.arange(0.02, 0.32, 0.02):
      c = np.round(c, 2)
      thread = Thread(target=scaleopt, args=(c, instance), daemon=True)
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

    with open(outfile, 'a') as f:
      for c in np.arange(0.02, 0.32, 0.02):
        c = np.round(c, 2)
        num = instance.split("/")[-1].split(".")[0]
        tempfile = os.path.join(f'out-{num}-{c}.txt')

        with open(tempfile, 'r') as g:
          shutil.copyfileobj(g, f)

        Path(tempfile).unlink(missing_ok=True)
    print(f'{instance} resolved')


##### TASK 3 ######
if args.task == 3:
  def regretopt(seed, instance):
    num = instance.split("/")[-1].split(".")[0]
    tempfile = os.path.join(dirname, f'out-{num}-{seed}.txt')
    Path(tempfile).unlink(missing_ok=True)
    hzs   = [100, 400, 1600, 6400, 25600, 102400]

    for hz in hzs:
      os.system(f"\
          python submission/bandit.py \
          --instance {instance} \
          --algorithm alg-t3 \
          --scale 0.22 \
          --randomSeed {seed} \
          --horizon {hz} >> {tempfile} \
        "
      )

  for file in os.listdir(instances):
    instance = os.path.join(instances, file)
    threads  = []

    for seed in range(50):
      thread = Thread(target=regretopt, args=(seed, instance), daemon=True)
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

    with open(outfile, 'a') as f:
      for seed in range(50):
        num = instance.split("/")[-1].split(".")[0]
        tempfile = os.path.join(f'out-{num}-{seed}.txt')

        with open(tempfile, 'r') as g:
          shutil.copyfileobj(g, f)

        Path(tempfile).unlink(missing_ok=True)
    print(f'{instance} resolved')


##### TASK 4 ######
if args.task == 4:
  def highopt(seed, instance):
    num = instance.split("/")[-1].split(".")[0]
    tempfile = os.path.join(dirname, f'out-{num}-{seed}.txt')
    Path(tempfile).unlink(missing_ok=True)
    hzs   = [100, 400, 1600, 6400, 25600, 102400]

    for hz in hzs:
      for th in [0.2, 0.6]:
        os.system(f"\
            python submission/bandit.py \
            --instance {instance} \
            --algorithm alg-t4 \
            --randomSeed {seed} \
            --threshold {th} \
            --horizon {hz} >> {tempfile} \
          "
        )

  for file in os.listdir(instances):
    instance = os.path.join(instances, file)
    threads  = []

    for seed in range(50):
      thread = Thread(target=highopt, args=(seed, instance), daemon=True)
      thread.start()
      threads.append(thread)

    for thread in threads:
      thread.join()

    with open(outfile, 'a') as f:
      for seed in range(50):
        num = instance.split("/")[-1].split(".")[0]
        tempfile = os.path.join(f'out-{num}-{seed}.txt')

        with open(tempfile, 'r') as g:
          shutil.copyfileobj(g, f)

        Path(tempfile).unlink(missing_ok=True)
    print(f'{instance} resolved')


if args.plot:
  os.system(f'python plot.py --file {outfile} --task {args.task}')
