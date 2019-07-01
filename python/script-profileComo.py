import cProfile, pstats
profilefile = 'profile2.dat'
if 11 < 3:
    prof = cProfile.Profile().runctx('import cma, numpy; ' +
                                     'res = cma.fmin(cma.fcts.rosen, numpy.ones(20), 1, ' +
                                     'verb_log=0, maxiter=100, bounds=[-1, 2.2], popsize=(200))',
                                     globals(), locals())
    s = pstats.Stats(prof)
elif 1 < 3:
    prof = cProfile.Profile().runctx("""
from comocmaes_many import CoMoCmaes
from problems import BiobjectiveConvexQuadraticProblem as problem
import numpy as np
dim = 5
num_kernels = 3
myproblem = problem(dim, name = "sphere")
fun = myproblem.objective_functions()
lbounds = -0*np.ones(dim)
rbounds = 1*np.ones(dim)
sigma0 = 0.2
refpoint = [1.1, 1.1]
budget = 1000*num_kernels
mymo = CoMoCmaes(fun,dim,sigma0,lbounds,rbounds,num_kernels,refpoint,budget,
                       name = myproblem.name)
mymo.run(budget)

    """, 
    globals(), locals())
    s = pstats.Stats(prof)
#elif 1 < 3:  # purecma
#    prof = cProfile.Profile().runctx('import cma.purecma as pcma; ' +
#                                     'pcma.fmin(pcma.ff.rosenbrock, 30 * [0.5], 0.5, maxfevals=2000)',
#                                     globals(), locals())
#    s = pstats.Stats(prof)
#
#elif 1 < 3:  # barecmaes
#    prof = cProfile.Profile().runctx('import barecmaes; ' +
#                                     'barecmaes.cmaes(barecmaes.frosenbrock, 20 * [0.5], 0.5, False)',
#                                     globals(), locals())
#    s = pstats.Stats(prof)
#
#elif 1 < 3:
#    cProfile.run('import cma, numpy; ' +
#                 'res = cma.fmin(cma.fcts.elli, numpy.ones(27), 1, ' +
#                 'verb_log=0, maxiter=200, CMAeigenmethod=1)',
#                 profilefile)
#    s = pstats.Stats(profilefile)
#
#else:
#    cProfile.run('import eigen, numpy; N=10;' +
#                 'M=numpy.random.rand(N**2).reshape(N,N);' +
#                 'eigen.tql2(numpy.dot(M,M))',
#                 profilefile)
#    s = pstats.Stats(profilefile)

s.sort_stats('cumulative').print_stats(15)
# s.sort_stats('time')
# s.print_stats()


# next: how get report on most expensive line 
