#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import cma
from cma import interfaces
from nondominatedarchive import NonDominatedList as NDL
from moarchiving import BiobjectiveNondominatedSortedList as BNDSL
import warnings
import cma.utilities.utils
import sys

class Sofomore(interfaces.OOOptimizer):
    
    """ 
    Sofomore framework for multiobjective optimization, with the 
    ask-and-tell interface.
    See: [Touré, Cheikh, et al. "Uncrowded Hypervolume Improvement: 
        COMO-CMA-ES and the Sofomore framework." 
        GECCO'19-Genetic and Evolutionary Computation Conference. 2019.].
        
    Calling Sequences
    =================

    - ``moes = Sofomore(list_of_solvers_instances, reference_point)``

    - ``moes = Sofomore(list_of_solvers_instances, reference_point, opts)``

    - ``moes = Sofomore(list_of_solvers_instances, 
                             reference_point).optimize(objective_fcts)``

    Arguments   
    =========
    `list_of_solvers_instances`
        list of instances of single-objective solvers.
        It's generally created via a factory function.
        Let's take the example of a factory function that returns cma-es 
        instances, called `get_cmas`. Then:
        - ``list_of_solvers_instances = get_cmas(11*[x0], sigma0)`` 
        creates a list of 11 cma-es instances of mean `x0` and step-size 
        `sigma0`.
        A single-objective solver instance must have the following
        attributes and methods:
            - `incumbent`: an attribute that gives the estimate of 
            the single-objective solver.
            - `objective_values`: an attribute that stores the objective 
            values of the incumbent.
            - `stop`: a method returning a dictionary representing the 
            termination status.
            - `ask`: generates new candidate solutions.
            - `tell`: passes the objective values and updates the states of the
            single-objective solvers.
                
    `reference_point`  
        reference point of the multiobjective optimization.
        Its default value is `None` but should be set by the user 
        beforehand to guarantee an optimal p-distribution convergence of 
        the Hypervolume indicator of `p` points towards the Pareto set/front.
        However, it can be changed dynamically by the user if needed.
        
    `options`
        options, a dictionary with optional settings related to the 
        Sofomore framework. It contains the following keys:
            - 'archive': if its value is `True`, tracks the non-dominated
            points among the points evaluated so far during the 
            optimization.
            Note that the archive will not interfere with the optimization 
            process.
            - 'update_order': a function from an ordered sequence to an ordered
            sequence, with the same set. It guides the order in which the
            kernels will be updated during the optimization.
 

    Main interface / usage
    ======================
    The interface is inherited from the generic `OOOptimizer`
    class (see also there). An object instance is generated from::
        
        list_of_solvers_instances = get_cmas(11*[x0], sigma0)
        moes = mo.Sofomore(list_of_solvers_instances,
                           reference_point = reference_point)

    The least verbose interface is via the optimize method::

         moes.optimize(objective_func)
TODO     res = moes.result

    More verbosely, the optimization is done using the
    methods `stop`, `ask`, and `tell`::

        
        while not moes.stop():
            solutions = moes.ask()
            objective_values = [[f(x) for f in fun] for x in solutions]
            moes.tell(solutions, objective_values)
TODO        moes.disp()
TODO        moes.result_pretty()

    where `ask` delivers new candidate solutions and `tell` updates
    the `optim` instance by passing the respective function values.

    Attributes and Properties
    =========================
    - `kernels`: initialized with `list_of_solvers_instances`, 
    and is the list of single-objective solvers.
    - `num_kernels`: length of `self.kernels`.
    - `options`: passed options.
    - `front`: list of non-dominated points among the incumbents of 
    `self.kernels`.
    - `archive`: list of non-dominated points among all points evaluated 
    so far.
    - `reference_point`: the current reference point.
    - `offspring`: list of tuples of the index of a kernel with its 
    corresponding candidate solutions, that we generally get with the cma's 
    `ask` method.
    - told_indices: the kernels' indices for which we will evaluate the 
    objective values of their incumbents, in the next call of the `tell` 
    method.
    Before the first call of `tell`, they are the indices of all the initial
    kernels (i.e. `range(self.num_kernels)`). And before another call of 
    `tell`, they are the indices of the kernels from which we have sampled new 
    candidate solutions during the penultimate `ask` method. 
    Note that we should call the `ask` method before any call of the `tell`
    method.
    """   
    def __init__(self,
               list_of_solvers_instances, # usally come from a factory function 
                                         #  creating single solvers' instances
               options = None, # keeping an archive, etc.
               reference_point = None,    
               ):
        """
        Initialization:
            - `list_of_solvers_instances` is a list of single-objective 
            solvers' instances
            - `options` is a dictionary updating the values of 
            `archive` and `update_order`, that responds respectfully to whether
            or not tracking an archive, and the order of update of the kernels.
            - The reference_point can be changed by the user after 
            initialization, by setting a value to `self.reference_point`.
        
        """
        assert len(list_of_solvers_instances) > 1
        self.kernels = list_of_solvers_instances
        self.num_kernels = len(self.kernels)
        for kernel in self.kernels:
            if not hasattr(kernel, 'objective_values'):
                kernel.objective_values = None
        self.reference_point = reference_point
        self.front = []
        update_order = lambda x: np.random.permutation(x)
        defopts = {'archive': False, 'update_order': update_order}
        if options is None:
            options = {}
        if isinstance(options, dict):
            defopts.update(options)
        else:
            warnings.warn("options should be either a dictionary or None.")
        self.options = defopts
        if self.options['archive']:
            self.archive = []
        self.offspring = []
        self.told_indices = range(self.num_kernels)
        
        seq = range(self.num_kernels)
        self._order = Sequence(self.options['update_order'], seq)()
        self.countiter = 0
        
    def ask(self, number_of_kernels = 1):
        """
        get the kernels' incumbents to be evaluated for the update of 
        `self.front` and sample new candidate solutions from 
        `number_of_kernels` kernels.
        The sampling is done by calling the `ask` method of the
        `cma.CMAEvolutionStrategy` class.
        The indices of the considered kernels' incumbents are given by the 
        `told_indices` attribute.
        
        Arguments
        ---------
        - `number_of_kernels`: the number of kernels where we sample 
        solutions from.
        
        Return
        ------
        The list of the kernels' incumbents to be evaluated, extended with a
        list of N-dimensional (N is the dimension of the search space) 
        candidate solutions generated from `number_of_kernels` kernels 
        to be evaluated.
    
        :See: the `ask` method from the class `cma.CMAEvolutionStrategy`,
            in `evolution_strategy.py` from the `cma` module.
        """
        if number_of_kernels == "all":
            number_of_kernels = self.num_kernels
        if number_of_kernels > self.num_kernels:
            warnings.warn('value larger than the number of kernels.')
        self.offspring = []
        res = [self.kernels[i].incumbent for i in self.told_indices]
        for ikernel in [next(self._order) for _ in range(number_of_kernels)]:
            kernel = self.kernels[ikernel]
            if not kernel.stop():
                offspring = kernel.ask()
                res.extend(offspring)
                self.offspring += [(ikernel, offspring)]
        return res
        
    def tell(self, solutions, objective_values):
        """
        pass objective function values to update respectfully: `self.front`, 
        the state variables of some kernels, `self.told_indices` and eventually 
        `self.archive`.
        Arguments
        ---------
        `solutions`
            list or array of points (of type `numpy.ndarray`), most presumably 
            before delivered by the `ask()` method.
        `objective_values`
            list of multiobjective function values (of type `list`)
            corresponding to the respective points in `solutions`.

        Details
        -------
        To update a kernel, `tell()` applies the kernel's `tell` method
        to the kernel's corresponding candidate solutions (offspring) along
        with the "changing" fitness `- self.front.hypervolume_improvement`.
        
        :See: 
            - the `tell` method from the class `cma.CMAEvolutionStrategy`,
            in `evolution_strategy.py` from the `cma` module.
            - the `hypervolume_improvement` method from the class
            `BiobjectiveNondominatedSortedList`, in the module `moarchiving.py`
            - the `hypervolume_improvement` method from the class
            `NonDominatedList`, in the module `nondominatedarchive.py`
        """
        if len(solutions) == 0: # when asking a terminated kernel for example
            return 

        NDA = BNDSL if len(objective_values[0]) == 2 else NDL
        for i in range(len(self.told_indices)):
            self.kernels[self.told_indices[i]].objective_values = objective_values[i]
        
        if self.reference_point is None:
            pass #write here the max among the kernel.objective_values       
        self.front = NDA([kernel.objective_values for kernel in self.kernels],
                         self.reference_point)
            
        start = len(self.told_indices) # position of the first offspring
        for ikernel, offspring in self.offspring:
            kernel = self.kernels[ikernel]
            fit = kernel.objective_values
            if fit in self.front: # i.e. if fit is not dominated and dominates 
                                  # the reference point
                self.front.remove(fit)
            hypervolume_improvements = [self.front.hypervolume_improvement(
                    point) for point in objective_values[start:start+len(offspring)]]
            self.front.add(fit) # in case num_kernels > 1
            start += len(offspring)
            kernel.tell(offspring, [-float(u) for u in hypervolume_improvements])
            try:
                kernel.logger.add()
            except:
                pass
            
        self.told_indices = [u for (u,v) in self.offspring]
       
        if self.options['archive']:
            if not self.archive:
                self.archive = NDA(objective_values, self.reference_point)
            else:
                self.archive.add_list(objective_values)
            
        self.countiter += 1

 
    def stop(self):
        """
        return a nonempty dictionary when all kernels stop, containing all the
        termination status. Therefore it's solely ... on the kernels' `stop`
        method, which also return dictionaries.
        Return
        ------
        For example with 5 kernels, stop should return either None, or a `dict`
        of the form:
            {0: dict0,
             1: dict1,
             2: dict2,
             3: dict2,
             4: dict4},
        where each index `i` is a key which value is the `dict` instance
        `self.kernels[i].stop()`
        """
        res = {}
        for i in range(len(self.kernels)):
            if self.kernels[i].stop():
                res[i] = self.kernels[i].stop()
            else:
                return False
        return res
            
        
    def turn_off(self, kernel):
        """
        inactivate `kernel`, assuming that it's an element of `self.kernels`,
        or an index in `range(self.num_kernels)`.
        When inactivated, `kernel` is no longer updated, it is ignored.
        However we do not remove it from `self.kernels`, meaning that `kernel`
        might still play a role, due to its eventual trace in `self.front`.
        """
        if kernel in self.kernels:
            try:
                kernel.opts['termination_callback'] += (lambda _: 'kernel turned off',)
            except (AttributeError, TypeError):
                warnings.warn("their is a problem with opts.")
        else:
            try:
                kernel = self.kernels[kernel]
                kernel.opts['termination_callback'] += (lambda _: 'kernel turned off',)
            except (AttributeError, TypeError):
                warnings.warn("their is a problem with opts.")

    def add(self, kernels):
        """
        add `kernels` of type `list` to `self.kernels` and update `self.front`
        and `self.num_kernels`.
        Generally, `kernels` are created from a factory function.
        If `kernels` is of length 1, the brackets can be omitted.
        """
        if not isinstance(kernels, list):
            kernels = [kernels]
        self.kernels += kernels
        self.num_kernels += len(kernels)
        
    def remove(self, kernels):
        """
        remove elements of the `kernels` (type `list`) that belong to
        `self.kernels`, and update `self.front` and
        `self.num_kernels` accordingly.
        If `kernels` is of length 1, the brackets can be omitted.
        """
        if not isinstance(kernels, list):
            kernels = [kernels]
        for kernel in kernels:
            if kernel in self.kernels:
                self.kernels.remove(kernel)
                if kernel.objective_values in self.front:
                    self.front.remove(kernel.objective_values)
            self.num_kernels -= 1

    
    @property
    def countevals(self):
        """
        """
        return sum(kernel.countevals for kernel in self.kernels)
    
    # The following methods 'disp_annotation' and 'disp' are from the 'cma'
    # module
    def disp_annotation(self):
        """print annotation line for `disp` ()"""
        self.has_been_called = True
        print('Iterat #Fevals   Hypervolume   axis ratios '
             '  sigmas   min&max stds\n'+'(median)'.rjust(42) +
             '(median)'.rjust(10) + '(median)'.rjust(12))
     #   try:
          #  sys.stdout.flush() : error in matlab:
          # Python Error: AttributeError: 'MexPrinter' object has no attribute 'flush'

      #  except AttributeError:
       #     pass
    def disp(self, modulo=None):
        """print current state variables in a single-line.

        Prints only if ``iteration_counter % modulo == 0``.

        :See also: `disp_annotation`.
        """
        if modulo is None:
            try:
                modulo = self.kernels[0].opts['verb_disp']
            except AttributeError:
                pass

        # console display

        if modulo:
            if not hasattr(self, 'has_been_called'):
                self.disp_annotation()

            if self.countiter > 0 and (self.stop() or self.countiter < 4
                              or self.countiter % modulo < 1):
                try:
                    print(' '.join((repr(self.countiter).rjust(5),
                                    repr(self.countevals).rjust(6),
                                    '%.15e' % (self.front.hypervolume),
                                    '%4.1e' % (np.median([kernel.D.max() / kernel.D.min()
                                               if not kernel.opts['CMA_diagonal'] or kernel.countiter > kernel.opts['CMA_diagonal']
                                               else max(kernel.sigma_vec*1) / min(kernel.sigma_vec*1) \
                                               for kernel in self.kernels])),
                                    '%6.2e' % (np.median([kernel.sigma for kernel in self.kernels])),
                                    '%6.0e' % (np.median([kernel.sigma * min(kernel.sigma_vec * kernel.dC**0.5) \
                                                         for kernel in self.kernels])),
                                    '%6.0e' % (np.median([kernel.sigma * max(kernel.sigma_vec * kernel.dC**0.5) \
                                                          for kernel in self.kernels]))
                                    )))
                except AttributeError:
                    pass
                    # if self.countiter < 4:
       #         try:
                  #  sys.stdout.flush() : error in matlab:
                  # Python Error: AttributeError: 'MexPrinter' object has no attribute 'flush'

         #       except AttributeError:
          #          pass
        return self


def get_cmas(x_starts, sigma_starts, inopts = None, number_created_kernels = 0):
    """
    produce `len(x_starts)` instances of type `cmaKernel`.
    """
    
    if x_starts is not None and len(x_starts):
        try:
            x_starts = x_starts.tolist()
        except:
            pass
        try:
            x_starts = [u.tolist() for u in x_starts]
        except:
            pass
        if not isinstance(x_starts[0], list):
            x_starts = [x_starts]
            
    kernels = []
    num_kernels = len(x_starts)
    if not isinstance(sigma_starts, list):
        sigma_starts = num_kernels * [sigma_starts]
    if inopts is None:
        inopts = {}
    list_of_opts = []
    if isinstance(inopts, list):
        list_of_opts = inopts
    else:
        list_of_opts = [dict(inopts) for _ in range(num_kernels)]
    
    for i in range(num_kernels):
        defopts = cma.CMAOptions()
        defopts.update({'verb_filenameprefix': str(number_created_kernels+i), 'conditioncov_alleviate': [np.inf, np.inf],
                    'verbose':-1, 'tolx':1e-9})
        if isinstance(list_of_opts[i], dict):
            defopts.update(list_of_opts[i])
            
        kernels += [CmaKernel(x_starts[i], sigma_starts[i], defopts)]
        
    return kernels

class CmaKernel(cma.CMAEvolutionStrategy):
    """
    inheriting from the `cma.CMAEvolutionStrategy` class, by slightly modifying
    the `stop` method, and adding the property `incumbent` and 
    the attribute `objective_values`.
    """
    def __init__(self, x0, sigma0, inopts=None):
        """
        """
        cma.CMAEvolutionStrategy.__init__(self, x0, sigma0, inopts)
        self.objective_values = None
    
    @property
    def incumbent(self):
        """
        """
        return self.boundary_handler.repair(self.mean)
    
    def stop(self, check=True, ignore_list=()):
        """
        'flat fitness' is ignored because it does not necessarily mean that a 
        termination criteria is met. For the `cigtab` bi-objective
        function for example, the Hypervolume is flat for a long period, 
        although the evolution is correctly occuring in the search space.
        """
        to_be_ignored = ignore_list + ('tolfun', 'tolfunhist', 
                                       'flat fitness', 'tolstagnation')
        
        return cma.CMAEvolutionStrategy.stop(self, check, ignore_list = to_be_ignored)





def order_generator(seq):
    """the generator for `randint_derandomized`
    code from the module cocopp, 
    in: cocopp.toolsstats._randint_derandomized_generator
    """
    size = len(seq)
    delivered = 0
    while delivered < size:
        for i in seq:
            delivered += 1
            yield i
            if delivered >= size:
                break
            
class Sequence(object):
    def __init__(self, permutation, seq):
        self.delivered = 0
        self.permutation = permutation
        self.seq = seq
        self.generator = order_generator(permutation(seq))
    def __call__(self):
        while True:
            for i in self.generator:
                self.delivered += 1
                yield i
                if self.delivered % len(self.seq) == 0:
                    self = Sequence(self.permutation, self.seq)


class RankPenalizedFitness:
    """compute f-values of infeasible solutions as rank_f-inverse(const + sum g-ranks).
    
    The inverse is computed by linear interpolation.
    
    Draw backs: does not support approaching the optimum from the infeasible domain.
    
    Infeasible solutions with valid f-value measurement could get a 1/2-scaled credit for their
    f-rank difference to the base f-value.
    """

    def __init__(self, f, g_list):
        self.f = f
        self.g_list = g_list
        # control parameters
        self.base_prctile = 0.2  # best f-value an infeasible solution can get
        self.g_scale = 1.01  # factor for g-ranks penalty
        self._debugging = True
        # internal state
        self.f_current_best = 0

    def __call__(self, X):
        """X is a list of solutions.
        
        Assumes that at least one solution does not return nan as f-value
        """
        # TODO: what to do if there is no f-value for a feasible solution
        f_values = [self.f(x) for x in X]
        g_ranks_list = []
        is_feasible = np.ones(len(X))
        for g in self.g_list:
            g_values = [g(x) for x in X]
            g_is_feas = np.asarray(g_values) <= 0
            is_feasible *= g_is_feas
            nb_feas = sum(g_is_feas)
            g_ranks = [g - nb_feas + 1 if g >= nb_feas else 0
                       for g in cma.utilities.utils.ranks(g_values)]  # TODO: this fails with nan-values
            if self._debugging: print(g_ranks)
            g_ranks_list.append(g_ranks)
        idx_feas = np.where(is_feasible)[0]
        # we could also add the distance to the best feasible solution as penalty on the median?
        # or we need to increase the individual g-weight with the number of iterations that no single
        #    feasible value was seen
        # change f-values of infeasible solutions
        sorted_feas_f_values = sorted(np.asarray(f_values)[idx_feas])
        try: self.f_current_best = sorted_feas_f_values[0]
        except IndexError: pass
        j0 = self.base_prctile * (len(idx_feas) - 1)
        for i in set(range(len(X))).difference(idx_feas):
            j = j0 + self.g_scale * (
                    sum(g_ranks[i] for g_ranks in g_ranks_list) - 1)  # -1 makes base a possible value
            assert j >= self.base_prctile * (len(idx_feas) - 1)
            # TODO: use f-value of infeasible solution if available?
            if 11 < 3 and np.isfinite(f_values[i]):
                self.gf_scale = 1 / 2
                j += self.gf_scale * (_interpolated_rank(f_values, f_values[i]) - 
                                      _interpolated_rank(f_values, f_values[j0]))  # TODO: filter f-values by np.isfinite
            j = max((j, 0))
            j1, j2 = int(j), int(np.ceil(j))
            f1 = self._f_from_index(sorted_feas_f_values, j1)
            f2 = self._f_from_index(sorted_feas_f_values, j2)
            # take weighted average fitness between index j and j+1
            f_values[i] = 0e-6 + (j - j1) * f2 + (j2 - j) * f1 if j2 > j1 else f1
        return f_values

    def _f_from_index(self, f_values, i):
        """`i` must be an integer but may be ``>= len(f_values)``"""
        imax = len(f_values) - 1
        if imax < 0:  # no feasible f-value
            return self.f_current_best + i
        return f_values[min((imax, i))] + max((i - imax, 0))
        
f = RankPenalizedFitness(lambda x: cma.ff.sphere(np.asarray(x)), [lambda x: x[0] > 0]) 
f(2 * [[1,2,3], [-1, 1, 10]] + [[1,2,3], [-1.1, 1, 10]] + 1 * [[-1, 1, 101]])
