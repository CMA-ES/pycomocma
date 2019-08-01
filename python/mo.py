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
        instances, called `get_cma`. Then:
        - ``list_of_solvers_instances = get_cma(11*[x0], sigma0)`` 
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
        
        list_of_solvers_instances = get_cma(11*[x0], sigma0)
        moes = mo.Sofomore(list_of_solvers_instances,
                           reference_point = reference_point)

    The least verbose interface is via the optimize method::

         moes.optimize(objective_func)
TODO     res = moes.result

    More verbosely, the optimization is done using the
    methods `stop`, `ask`, and `tell`::

        
        while not moes.stop():
            solutions = moes.ask()
            objective_values = [fitness(x) for x in solutions]
            moes.tell(solutions, objective_values)
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
        update_order = lambda x: x
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
        order = Sequence(self.options['update_order'](range(self.num_kernels)))()
        res = [self.kernels[i].incumbent for i in self.told_indices]
        for ikernel in [order.__next__() for _ in range(number_of_kernels)]:
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
            kernel.opts['termination_callback'] = lambda _: 'kernel turned off'
        else:
            try:
                kernel = self.kernels[kernel]
                kernel.opts['termination_callback'] = lambda _: 'kernel turned off'
            except:
                pass
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

def get_cma(x_starts, sigma_starts, inopts = None, number_created_kernels = 0):
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
        return self.mean
    
    def stop(self, check=True, ignore_list=()):
        """
        """
        to_be_ignored = ignore_list + ('tolfun', 'tolfunhist', 'flat fitness', 'tolstagnation')
        return cma.CMAEvolutionStrategy.stop(self, check, ignore_list = to_be_ignored)

def _randint_derandomized_generator(order):
    """the generator for `randint_derandomized`
    code from the module cocopp, in: cocopp.toolsstats._randint_derandomized_generator
    """
    size = len(order)
    delivered = 0
    while delivered < size:
        for i in order:
            delivered += 1
            yield i
            if delivered >= size:
                break
            
class Sequence(object):
    def __init__(self, seq, order = lambda x: np.random.permutation(x)):
        self.generator = _randint_derandomized_generator(order(seq))
        self.delivered = 0
    def __call__(self):
        while True:
            for i in self.generator:
                self.delivered += 1
                yield i
        
        
        
        
        
