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
            the solver.
            - `objective_values`: an attribute that store the objective 
            values of the incumbent.
            - `stop`: a method returning a dictionary representing the 
            termination status.
            - `ask`: generate new candidate solutions.
            - `tell`: pass the objective values and update the states of the
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
            - 'update_order': the order in which the kernels will be updated.
 

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
            X = moes.ask()
            F = [fitness(x) for x in X]
            moes.tell(X, F)
            moes.tell(solutions, F)
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
    self.kernels.
    - `archive`: list of non-dominated points among all points evaluated 
    so far.
    - `reference_point`: the current reference point.
    - `offspring`: list of tuples of the index of a solver and the generated
    candidate solutions, that we generally get with the cma's `ask` method.
    - asked-indices: the kernels' indices updated during the
    last `ask-and-tell` call.
        
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
            pass # TODO raise a warning at least here.
        self.options = defopts
        if self.options['archive']:
            self.archive = []
        self.offspring = []
        self.update_order = Sequence(self.num_kernels)()
        self.told_indices = range(self.num_kernels)
        
    def ask(self, num_kernels = 1):
        """
        get/sample new feasible candidate solutions, by constantly calling the
        `ask` method of the `cma.CMAEvolutionStrategy` class.
        
        Arguments
        ---------
        - `num_kernels`: the number of kernels that we sample solutions from.
        
        Return
        ------
        A list of N-dimensional (N is the dimension of the search space) 
        candidate solutions generated from `num_kernels` kernels 
        to be evaluated."""
        
        if num_kernels == "all":
            num_kernels = self.num_kernels
        if num_kernels > self.num_kernels:
            warnings.warn('value larger than the number of kernels.')
        self.offspring = []
        res = [self.kernels[i].incumbent for i in self.told_indices]
        for ikernel in [self.update_order.__next__() for _ in range(num_kernels)]:
            kernel = self.kernels[ikernel]
            if not kernel.stop():
                offspring = kernel.ask()
                res.extend(offspring)
                self.offspring += [(ikernel, offspring)]
        return res
        
    def tell(self, X, F):
        """
        """
        if len(X) == 0: # when asking a terminated kernel for example
            return 

        NDA = BNDSL if len(F[0]) == 2 else NDL
        for i in range(len(self.told_indices)):
            self.kernels[self.told_indices[i]].objective_values = F[i]
        
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
                    point) for point in F[start:start+len(offspring)]]
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
                self.archive = NDA(F, self.reference_point)
            else:
                self.archive.add_list(F)

 
    def stop(self):
        """
        return a nonempty dictionary when all kernels stop, containing all the
        termination status

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
        turn off ‘kernel‘ in self, when ‘kernel‘ is in self.
        """
        if kernel in self.kernels:
    #        kernel.stop()['turn_off'] = True 
            kernel.opts['termination_callback'] = lambda _: 'kernel turned off'
    
    def add(self, kernels):
        """
        add kernel to self.
        """
        if not isinstance(kernels, list):
            kernels = [kernels]
        self.kernels += kernels
        self.num_kernels += len(kernels)
    def remove(self, kernel):
        """
        remove ‘kernel‘ if it's in self.
        """
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

def _randint_derandomized_generator(low, high=None, size=None):
    """the generator for `randint_derandomized`
    code from the module cocopp, in: cocopp.toolsstats._randint_derandomized_generator
    """
    if high is None:
        low, high = 0, low
    if size is None:
        size = high
    delivered = 0
    while delivered < size:
        for randi in np.random.permutation(high - low):
            delivered += 1
            yield low + randi
            if delivered >= size:
                break
            
class Sequence(object):
    def __init__(self, max_val, generator=_randint_derandomized_generator):
        self.max_val = max_val
        self.generator = generator
        self.delivered = 0
    def __call__(self):
        while True:
            for randi in self.generator(self.max_val):
                self.delivered += 1
                yield randi
        
        
        
        
        
