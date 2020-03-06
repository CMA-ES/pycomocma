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
import cma
from cma_extensions.constraints_handler import ConstrainedSurrogateFunction
ff = ConstrainedSurrogateFunction(cma.ff.schaffer, expo=1.5)
# ff = cma.ff.schaffer
xopt, es = cma.fmin2(ff, 10 * [1], 1, {'CMA_active': True},
                        restarts=1)
    """, 
    globals(), locals())
    s = pstats.Stats(prof)
elif 1 < 3:  # purecma
    prof = cProfile.Profile().runctx('import cma.purecma as pcma; ' +
                                     'pcma.fmin(pcma.ff.rosenbrock, 30 * [0.5], 0.5, maxfevals=2000)',
                                     globals(), locals())
    s = pstats.Stats(prof)

elif 1 < 3:  # barecmaes
    prof = cProfile.Profile().runctx('import barecmaes; ' +
                                     'barecmaes.cmaes(barecmaes.frosenbrock, 20 * [0.5], 0.5, False)',
                                     globals(), locals())
    s = pstats.Stats(prof)

elif 1 < 3:
    cProfile.run('import cma, numpy; ' +
                 'res = cma.fmin(cma.fcts.elli, numpy.ones(27), 1, ' +
                 'verb_log=0, maxiter=200, CMAeigenmethod=1)',
                 profilefile)
    s = pstats.Stats(profilefile)

else:
    cProfile.run('import eigen, numpy; N=10;' +
                 'M=numpy.random.rand(N**2).reshape(N,N);' +
                 'eigen.tql2(numpy.dot(M,M))',
                 profilefile)
    s = pstats.Stats(profilefile)

s.sort_stats('cumulative').print_stats(10)
# s.sort_stats('time')
# s.print_stats()


# next: how get report on most expensive line 
