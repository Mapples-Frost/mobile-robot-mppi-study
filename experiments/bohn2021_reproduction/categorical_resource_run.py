"""Bound eager replay allocation by the total number of transitions in this run."""
import sys
from pathlib import Path
import categorical_run as runner
from run import write


if __name__ == '__main__':
    steps = int(sys.argv[sys.argv.index('--steps') + 1])
    original = runner.ReplayBuffer
    runner.ReplayBuffer = lambda capacity: original(min(capacity, steps))
    folder = Path(sys.argv[sys.argv.index('--out') + 1])
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / 'replay_allocation.json', {'configured_capacity': 1000000,
          'allocated_capacity': min(1000000, steps), 'run_transitions': steps,
          'evictions_before_run_end': 0, 'resource_amendment': 'See parent resource_amendment.json'})
    runner.main()
