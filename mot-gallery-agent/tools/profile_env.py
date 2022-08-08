import cProfile
from pstats import Stats

pr = cProfile.Profile()
pr.enable()

### PROFILING START ###

from tools.manual_test_sequential import run_sequential_env

run_sequential_env(greedy=True)

### PROFILING END ###

pr.disable()
with open('profiling_stats.txt', 'w') as stream:
    stats = Stats(pr, stream=stream)
    stats.strip_dirs()
    stats.sort_stats('time')
    stats.dump_stats('.prof_stats')  # snakeviz .prof_stats
    stats.print_stats()
