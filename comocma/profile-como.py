import cProfile, pstats
profilefile = 'profile3.dat'

cProfile.run("""
import cma, como
import numpy as np
num_kernels = 1  # number of single-objective solvers (population size)
sigma0 = 2
dimension = 5
list_of_solvers = como.get_cmas(-5 + 10 * np.random.rand(num_kernels, dimension),
                                sigma0,  # inopts=cmaopts
                                {'tolx': 10 ** -4, 
                                 'tolfunrel': 1e-2,
                                 'maxiter': 1e9}
                                )  # produce `num_kernels cma instances`
moes = como.Sofomore(list_of_solvers, opts={'archive': False,
                                     #'restart': como.random_restart_kernel},
                                     'restart': como.best_chv_restart_kernel,
                                     #'restart': como.best_chv_or_random_restart_kernel,
                                     'continue_stopped_kernel': False,
                                     'random_restart_on_domination': False},
                                     reference_point = [11,11])
fitness = como.FitFun(cma.ff.sphere, lambda x: cma.ff.sphere(x-1))
iter = 1
while not moes.stop() and iter < 1000:
    solutions = moes.ask()
    objective_values = [fitness(x) for x in solutions]
    moes.tell(solutions, objective_values)
    iter = iter + 1
""",
profilefile)
s = pstats.Stats(profilefile)
from pstats import SortKey
s.sort_stats('tottime').print_stats(100)

s.sort_stats('cumulative').print_stats(10)
# s.sort_stats('time')
# s.print_stats()


# next: how get report on most expensive line
