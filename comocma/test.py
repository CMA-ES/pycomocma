"""Run doctests like ``python -m comocma.test``.

Test application of boundaries:

>>> import cma, comocma

>>> dimension = 10  # dimension of the search space
>>> num_kernels = 5 # number of single-objective solvers (number of points on the front)
>>> sigma0 = 0.2    # initial step-sizes

>>> opts = {"bounds": [-5, 5]}

>>> list_of_solvers = comocma.get_cmas(num_kernels * [dimension * [0]], sigma0, inopts = opts) 
>>> moes = comocma.Sofomore(list_of_solvers, [11,11]) 

>>> fitness = comocma.FitFun(cma.ff.sphere, lambda x: cma.ff.sphere(x-1)) 

>>> while not moes.stop():
...     solutions = moes.ask("all")
...     objective_values = [fitness(x) for x in solutions]
...     moes.tell(solutions, objective_values)
...     # moes.disp()
...     # moes.logger.add()
>>> assert float(moes.best_hypervolume_pareto_front) > 95, float(moes.best_hypervolume_pareto_front)

"""

def main():
    import doctest
    import comocma  # leads to circular imports when on top
    # import inspect

    doctest.ELLIPSIS_MARKER = '***' # to be able to ignore an entire output, 
        # putting the default '...' doesn't work for that.
    # print('doctesting `comocma`')
    print(doctest.testmod(comocma.como))
    print(doctest.testmod(comocma.test))

if __name__ == "__main__":
    main()