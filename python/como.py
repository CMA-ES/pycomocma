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
import os
from sofomore_logger import SofomoreDataLogger
#import sys

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
        
    `opts`
        opts, a dictionary with optional settings related to the 
        Sofomore framework. It contains the following keys:
            - 'archive': default value is `True`. 
            If its value is `True`, tracks the non-dominated
            points among the points evaluated so far during the 
            optimization.
            Note that the archive will not interfere with the optimization 
            process.
            - 'update_order': default value is `None`.
            A real-valued function that takes an int instance as argument.
            It guides the order in which the kernels will be updated during 
            the optimization.
 

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
            moes.disp()
TODO        moes.result_pretty()

    where `ask` delivers new candidate solutions and `tell` updates
    the `optim` instance by passing the respective function values.

    Attributes and Properties
    =========================
    - `kernels`: initialized with `list_of_solvers_instances`, 
    and is the list of single-objective solvers.
    - `num_kernels`: length of `self.kernels`.
    - `opts`: passed options.
    - `pareto_front`: list of non-dominated points among the incumbents of 
    `self.kernels`.
    - `archive`: list of non-dominated points among all points evaluated 
    so far.
    - `reference_point`: the current reference point.
    - `offspring`: list of tuples of the index of a kernel with its 
    corresponding candidate solutions, that we generally get with the cma's 
    `ask` method.
    - _told_indices: the kernels' indices for which we will evaluate the 
    objective values of their incumbents, in the next call of the `tell` 
    method.
    Before the first call of `tell`, they are the indices of all the initial
    kernels (i.e. `range(self.num_kernels)`). And before another call of 
    `tell`, they are the indices of the kernels from which we have sampled new 
    candidate solutions during the penultimate `ask` method. 
    Note that we should call the `ask` method before any call of the `tell`
    method.
    - `key_sort_indices`: default value is `self.opts['update_order']`.
    It is a function used as a key to sort some kernels' indices, in order to
    select the first indices during the call of the `ask` method.
    """   
    def __init__(self,
               list_of_solvers_instances, # usally come from a factory function 
                                         #  creating single solvers' instances
               opts = None, # keeping an archive, etc.
               reference_point = None,    
               ):
        """
        Initialization:
            - `list_of_solvers_instances` is a list of single-objective 
            solvers' instances
            - `opts` is a dictionary updating the values of 
            `archive` and `update_order`, that responds respectfully to whether
            or not tracking an archive, and the order of update of the kernels.
            - The reference_point can be changed by the user after 
            initialization, by setting a value to `self.reference_point`.
        
        """
        assert len(list_of_solvers_instances) > 0
        self.kernels = list_of_solvers_instances
        self.num_kernels = len(self.kernels)
        for kernel in self.kernels:
            if not hasattr(kernel, 'objective_values'):
                kernel.objective_values = None
        self.reference_point = reference_point
        self.pareto_front = []
        defopts = {'archive': True, 'verb_filenameprefix': 'outsofomore' + os.sep, 
                   'verb_log': 1, 'verb_disp': 100, 'update_order': None}
        if opts is None:
            opts = {}
        if isinstance(opts, dict):
            defopts.update(opts)
        else:
            warnings.warn("options should be either a dictionary or None.")
        self.opts = defopts
        self.active_archive = self.opts['archive']
        if self.active_archive:
            self.archive = []
        self.nda = None # the method for nondominated archiving
        self.offspring = []
        self._told_indices = range(self.num_kernels)
        
        #self._order = Sequence(self.options['update_order'], seq)() # generator
        self.key_sort_indices = self.opts['update_order']
        self.countiter = 0
        self._remaining_indices_to_ask = range(self.num_kernels) # where we look when calling `ask`
        self.logger = SofomoreDataLogger(self.opts['verb_filenameprefix'],
                                                     modulo=self.opts['verb_log']).register(self)
        self.best_hypervolume_pareto_front = 0.0
        self.epsilon_hypervolume_pareto_front = 0.1 # the minimum positive convergence gap
        
        self._ratio_nondom_offspring_incumbent = self.num_kernels * [0]
    def __iter__(self):
        """
        making `self` iterable. 
        Future work: it would be interesting to make it subscriptable.
        """
        return iter(self.kernels)
        
#        
#    def _modulo(self, number_asks):
#        """
#        `number_asks` is an int.
#        
#        returns the list `[self._start_ask % self.num_kernels, ..., 
#        (self._start_ask + number_asks - 1) % self.num_kernels]`.
#        
#        Designed to be used in the `ask` method:
#        self._modulo(number_asks) returns `number asks` successive integers, 
#        starting from `self._start_ask`, and whenever `self.num_kernels` is 
#        reached, we replace it by `0` and continue the sequence from `0`: 
#        it's a torus.
#        
#        Example:
#        --------    
#        If self.num_kernels = 5 and self._start_ask = 3:
#        self._modulo(5) = [3, 4, 0, 1, 2]
#        """
#        res = []
#        assert number_asks > 0
#        for k in range(number_asks):
#            res += [(self._start_ask + k) % self.num_kernels]
#        return res
        
        
    def ask(self, number_asks = 1):
        """
        get the kernels' incumbents to be evaluated for the update of 
        `self.pareto_front` and sample new candidate solutions from 
        `number_asks` kernels.
        The sampling is done by calling the `ask` method of the
        `cma.CMAEvolutionStrategy` class.
        The indices of the considered kernels' incumbents are given by the 
        `_told_indices` attribute.
        
        To get the `number_asks` kernels, we use the function `self.key_sort_indices` as
        a key to sort `self._remaining_indices_to_ask` (which is the list of
        kernels' indices wherein we choose the first `number_asks` elements.
        And if `number_asks` is larger than `len(self._remaining_indices_to_ask)`,
        we select the list `self._remaining_indices_to_ask` extended with the  
        first `number_asks - len(self._remaining_indices_to_ask)` elements
        of `range(self.num_kernels)`, sorted with `self.key_sort_indices` as key.

        Arguments
        ---------
        - `number_asks`: the number of kernels where we sample 
        solutions from, it's of type int and is smaller or equal to `self.num_kernels`
        
        Return
        ------
        The list of the kernels' incumbents to be evaluated, extended with a
        list of N-dimensional (N is the dimension of the search space) 
        candidate solutions generated from `number_asks` kernels 
        to be evaluated.
    
        :See: the `ask` method from the class `cma.CMAEvolutionStrategy`,
            in `evolution_strategy.py` from the `cma` module.
            
        TODO: only ask `active` kernels
        """
        if number_asks == "all":
            number_asks = self.num_kernels
        assert number_asks > 0
        if number_asks > self.num_kernels:
            number_asks = self.num_kernels
            warnings.warn("value larger than the number of kernels {}. ".format(
                    self.num_kernels) + "Set to {}.".format(self.num_kernels))
        self.offspring = []
        res = [self.kernels[i].incumbent for i in self._told_indices]
      
        sorted_indices = sorted(self._remaining_indices_to_ask, key = self.key_sort_indices)
        indices_to_ask = []
        remaining_indices = []
        if number_asks <= len(sorted_indices):
            indices_to_ask = sorted_indices[:number_asks]
            remaining_indices = sorted_indices[number_asks:]
        else:
            val = number_asks - len(sorted_indices)
            indices_to_ask = sorted_indices
            sorted_indices = sorted(range(self.num_kernels), key = self.key_sort_indices)
            indices_to_ask += sorted_indices[:val]
            remaining_indices = sorted_indices[val:]
            
        for ikernel in indices_to_ask:
            kernel = self.kernels[ikernel]
            if not kernel.stop():
                offspring = kernel.ask()
                res.extend(offspring)
                self.offspring += [(ikernel, offspring)]
        self._remaining_indices_to_ask = remaining_indices
        return res
               
        
        
#    def askkk(self, number_asks = 1):
#        """
#        get the kernels' incumbents to be evaluated for the update of 
#        `self.pareto_front` and sample new candidate solutions from 
#        `number_asks` kernels.
#        The sampling is done by calling the `ask` method of the
#        `cma.CMAEvolutionStrategy` class.
#        The indices of the considered kernels' incumbents are given by the 
#        `_told_indices` attribute.
#        
#        Arguments
#        ---------
#        - `number_asks`: the number of kernels where we sample 
#        solutions from.
#        
#        Return
#        ------
#        The list of the kernels' incumbents to be evaluated, extended with a
#        list of N-dimensional (N is the dimension of the search space) 
#        candidate solutions generated from `number_asks` kernels 
#        to be evaluated.
#    
#        :See: the `ask` method from the class `cma.CMAEvolutionStrategy`,
#            in `evolution_strategy.py` from the `cma` module.
#        """
#        if number_asks == "all":
#            number_asks = self.num_kernels
#        if number_asks > self.num_kernels:
#            warnings.warn('value larger than the number of kernels.')
#        self.offspring = []
#        res = [self.kernels[i].incumbent for i in self._told_indices]
#        for ikernel in [next(self._order) for _ in range(number_asks)]:
#            kernel = self.kernels[ikernel]
#            if not kernel.stop():
#                offspring = kernel.ask()
#                res.extend(offspring)
#                self.offspring += [(ikernel, offspring)]
#        return res
        
    def tell(self, solutions, objective_values, constraints_values = []):
        """
        pass objective function values to update respectfully: `self.pareto_front`, 
        the state variables of some kernels, `self._told_indices` and eventually 
        `self.archive`.
        Arguments
        ---------
        `solutions`
            list or array of points (of type `numpy.ndarray`), most presumably 
            before delivered by the `ask()` method.
        `objective_values`
            list of multiobjective function values (of type `list`)
            corresponding to the respective points in `solutions`.
        `constraints_values`
            list of list of constraint values: each element is a list containing
            the values of one constraint function, that are obtained by evaluation
            on `solutions`.
            

        Details
        -------
        To update a kernel, `tell()` applies the kernel's `tell` method
        to the kernel's corresponding candidate solutions (offspring) along
        with the "changing" fitness `- self.pareto_front.hypervolume_improvement`.
        
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
        if self.nda is None:
            self.nda = BNDSL if len(objective_values[0]) == 2 else NDL
        for i in range(len(self._told_indices)):
            self.kernels[self._told_indices[i]].objective_values = objective_values[i]
        
        if self.reference_point is None:
            pass #write here the max among the kernel.objective_values       
        self.pareto_front = self.nda([kernel.objective_values for kernel in self.kernels],
                         self.reference_point)
            
        start = len(self._told_indices) # position of the first offspring
        for ikernel, offspring in self.offspring:
            kernel = self.kernels[ikernel]
            fit = kernel.objective_values
            if fit in self.pareto_front: # i.e. if fit is not dominated and dominates 
                                  # the reference point
                self.pareto_front.remove(fit)
            hypervolume_improvements = [self.pareto_front.hypervolume_improvement(
                    point) for point in objective_values[start:start+len(offspring)]]
            self.pareto_front.add(fit) # in case num_kernels > 1
            
            g_values = [constraint[start:start+len(offspring)] \
                        for constraint in constraints_values]
            penalized_f_values = RankPenalizedFitness([-float(u) for u in 
                                hypervolume_improvements], g_values)
            kernel.tell(offspring, penalized_f_values())
#            kernel.tell(offspring, [-float(u) for u in hypervolume_improvements])
            try:
                kernel.logger.add()
            except:
                pass
            kernel.last_offspring_f_values = objective_values[start:start+len(offspring)]
            
            start += len(offspring)
            
        self._told_indices = [u for (u,v) in self.offspring]
        current_hypervolume = self.pareto_front.hypervolume
        epsilon = abs(current_hypervolume - self.best_hypervolume_pareto_front)
        if epsilon:
            self.epsilon_hypervolume_pareto_front = min(self.epsilon_hypervolume_pareto_front, 
                                                        epsilon)
        self.best_hypervolume_pareto_front = max(self.best_hypervolume_pareto_front,
                                                 current_hypervolume)
        if self.active_archive:
            if not self.archive:
                self.archive = self.nda(objective_values, self.reference_point)
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
        for i in range(self.num_kernels):
            if self.kernels[i].stop():
                res[i] = self.kernels[i].stop()
            else:
                return False
        return res
            
    @property 
    def termination_status(self):
        """
        return a dictionary of the current termination states of the kernels.
        
        """
        res = {}
        for i in range(self.num_kernels):
            res[i] = self.kernels[i].stop()
        return res
    
    @property
    def median_stds(self):
        """
        """
        tab = []
        res = []
        for kernel in self.kernels:
            conv = np.sqrt(kernel.dC)
            vec = []
            for i in range(len(kernel.pc)):
                xi = max(kernel.sigma_vec*kernel.pc[i], kernel.sigma_vec*conv[i])
                vec += [kernel.sigma * xi / kernel.sigma0]
            tab += [sorted(vec)]
        for i in range(len(tab[0])):
            vec = [u[i] for u in tab]
            res += [np.median(vec)]
        return res

    @property
    def max_max_stds(self):
        """
        """
        res = 0.0
        for kernel in self.kernels:
            conv = np.sqrt(kernel.dC)
            vec = []
            for i in range(len(kernel.pc)):
                xi = max(kernel.sigma_vec*kernel.pc[i], kernel.sigma_vec*conv[i])
                vec += [kernel.sigma * xi / kernel.sigma0]
            res = max(res, max(vec))
        return res    
    
    def inactivate(self, kernel):
        """
        inactivate `kernel`, assuming that it's an element of `self.kernels`,
        or an index in `range(self.num_kernels)`.
        When inactivated, `kernel` is no longer updated, it is ignored.
        However we do not remove it from `self.kernels`, meaning that `kernel`
        might still play a role, due to its eventual trace in `self.pareto_front`.
    
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
                
    def activate(self, kernel):
        """
        activate `kernel` when it was inactivated beforehand. Otherwise 
        it remains quiet.
        
        We expect the kernel's `stop` method in interest to look like:
        kernel.stop() = {'callback': ['kernel turned off']}
        """
        raise NotImplementedError
        
        

    def add(self, kernels):
        """
        add `kernels` of type `list` to `self.kernels` and update `self.pareto_front`
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
        `self.kernels`, and update `self.pareto_front` and
        `self.num_kernels` accordingly.
        If `kernels` is of length 1, the brackets can be omitted.
        """
        if not isinstance(kernels, list):
            kernels = [kernels]
        for kernel in kernels:
            if kernel in self.kernels:
                self.kernels.remove(kernel)
                if kernel.objective_values in self.pareto_front:
                    self.pareto_front.remove(kernel.objective_values)
            self.num_kernels -= 1

    @property
    def pareto_set(self):
        """
        return the estimated Pareto set of the algorithm, among the kernels'
        incumbents.
        It's the pre-image of `self.pareto_front`.
        """
        return [kernel.incumbent for kernel in self.kernels if \
                kernel.objective_values in self.pareto_front]

    @property
    def ratio_inactive(self):
        """
        return the ratio of inactive kernels among all kernels.
        """
        ratio = 0
        for kernel in self.kernels:
            if kernel.stop():
                ratio += 1/self.num_kernels
        return ratio
    @property
    def countevals(self):
        """
        return the number of function evaluations during the optimization.
        Note that the single-objective solvers must have an attriburte 
        `countevals`.
        """
        return sum(kernel.countevals for kernel in self.kernels)
    
    # The following methods 'disp_annotation' and 'disp' are from the 'cma'
    # module
    def disp_annotation(self):
        """
        copy-pasted from `cma.evolution_strategy`.
        print annotation line for `disp` ()"""
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
        """
        copy-pasted from `cma.evolution_strategy`.
        print current state variables in a single-line.
        copy-pasted from `cma.evolution_strategy` module

        Prints only if ``iteration_counter % modulo == 0``.

        :See also: `disp_annotation`.
        """
        if modulo is None:
            modulo = self.opts['verb_disp']

        # console display

        if modulo:
            if not hasattr(self, 'has_been_called'):
                self.disp_annotation()

            if self.countiter > 0 and (self.stop() or self.countiter < 4
                              or self.countiter % modulo < 1):
                try:
                    print(' '.join((repr(self.countiter).rjust(5),
                                    repr(self.countevals).rjust(6),
                                    '%.15e' % (self.pareto_front.hypervolume),
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
    
    # repairing the initial values:
    for i in range(len(x_starts)):
        try:
            bounds_transform = cma.constraints_handler.BoundTransform(list_of_opts[i]['bounds'])        
            x_starts[i] = bounds_transform.repair(x_starts[i])
        except KeyError:
            pass
    
    for i in range(num_kernels):
        defopts = cma.CMAOptions()
        defopts.update({'verb_filenameprefix': 'cma_kernels' + os.sep + 
                        str(number_created_kernels+i), 'conditioncov_alleviate': [np.inf, np.inf],
                    'verbose': -1, 'tolx': 1e-6}) # default: normalize 'tolx' value. 
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
        Arguments
        =========
        `x0`
            initial solution, starting point. `x0` is given as "phenotype"
            which means, if::
    
                opts = {'transformation': [transform, inverse]}
    
            is given and ``inverse is None``, the initial mean is not
            consistent with `x0` in that ``transform(mean)`` does not
            equal to `x0` unless ``transform(mean)`` equals ``mean``.
        `sigma0`
            initial standard deviation.  The problem variables should
            have been scaled, such that a single standard deviation
            on all variables is useful and the optimum is expected to
            lie within about `x0` +- ``3*sigma0``. See also options
            `scaling_of_variables`. Often one wants to check for
            solutions close to the initial point. This allows,
            for example, for an easier check of consistency of the
            objective function and its interfacing with the optimizer.
            In this case, a much smaller `sigma0` is advisable.
        `inopts`
            options, a dictionary with optional settings,
            see class `cma.CMAOptions`.
            """
        cma.CMAEvolutionStrategy.__init__(self, x0, sigma0, inopts)
        self.objective_values = None # the objective value of self's incumbent
        # (see below for definition of incumbent)
        self.last_offspring_f_values = None # the fvalues of its offspring
        # used in the last call of `tell`.  
    
    @property
    def incumbent(self):
        """
        it gives the 'repaired' mean of a cma-es. For a problem with bound
        constraints, `self.incumbent` in inside the bounds.
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

class FitFun:
    """
    Define a callable multiobjective function from single objective ones.
    Example:
        fitness = FitFun(cma.ff.sphere, lambda x: cma.ff.sphere(x-1)).
    """
    def __init__(self, *args):
        self.callables = args
    def __call__(self, x):
        return [f(x) for f in self.callables]

#class Order(object):
#    """
#    `Order(optimizer)` is a function that takes an index `i` as argument, and returns
#    the opposite contributing hypervolume of `optimizer.kernels[i].objective_values`
#    into `optimizer.pareto_front`.
#    Example:
#        list_of_solvers = mo.get_cmas(num_kernels * [dimension * [0.3]], 0.2)
#        moes = mo.Sofomore(list_of_solvers, reference_point = [11,11])
#        moes.order = Order(moes)
#
#    """
#    def __init__(self, optimizer):
#        self.optimizer = optimizer
#    def __call__(self,i):
#        if self.optimizer.kernels[i].objective_values not in self.optimizer.pareto_front:
#            # meaning that the point is not nondominated
#            return 0
#        else: # the point is nondominated: the (opposite) contributing hypervolume is a non zero value
#            index = self.optimizer.pareto_front.bisect_left(self.optimizer.kernels[i].objective_values)
#            return - self.optimizer.pareto_front.contributing_hypervolume(index)

class RankPenalizedFitness:
    """compute f-values of infeasible solutions as rank_f-inverse(const + sum g-ranks).
    
    The inverse is computed by linear interpolation.
    
    Draw backs: does not support approaching the optimum from the infeasible domain.
    
    Infeasible solutions with valid f-value measurement could get a 1/2-scaled credit for their
    f-rank difference to the base f-value.
    """

    def __init__(self, f_values, g_list_values):
        self.f_values = f_values
        self.g_list_values = g_list_values
        # control parameters
        self.base_prctile = 0.2  # best f-value an infeasible solution can get
        self.g_scale = 1.01  # factor for g-ranks penalty
        self._debugging = False
        # internal state
        self.f_current_best = 0

    def __call__(self):
        """
        Assumes that at least one solution does not return nan as f-value
        """
        # TODO: what to do if there is no f-value for a feasible solution
     #   f_values = [self.f(x) for x in X]
        f_values = self.f_values
        g_ranks_list = []
        is_feasible = np.ones(len(f_values))
   #     for g in self.g_list:
        for g_values in self.g_list_values:
    #        g_values = [g(x) for x in X]         
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
        #         for i in set(range(len(X))).difference(idx_feas):
        for i in set(range(len(f_values))).difference(idx_feas):
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
        
#f = RankPenalizedFitness(lambda x: cma.ff.sphere(np.asarray(x)), [lambda x: x[0] > 0]) 
#f(2 * [[1,2,3], [-1, 1, 10]] + [[1,2,3], [-1.1, 1, 10]] + 1 * [[-1, 1, 101]])
        

#def order_generator(seq):
#    """the generator for `randint_derandomized`
#    code from the module cocopp, 
#    in: cocopp.toolsstats._randint_derandomized_generator
#    """
#    size = len(seq)
#    delivered = 0
#    while delivered < size:
#        for i in seq:
#            delivered += 1
#            yield i
#            if delivered >= size:
#                break
#            
#class Sequence(object):
#    """
#    TODO: docstring + comments to be done.
#    """
#    def __init__(self, permutation, seq):
#        self.delivered = 0
#        self.permutation = permutation
#        self.seq = seq
#        self.generator = order_generator(permutation(seq))
#    def __call__(self):
#        while True:
#            for i in self.generator:
#                self.delivered += 1
#                yield i
#                if self.delivered % len(self.seq) == 0:
#                    self = Sequence(self.permutation, self.seq)
