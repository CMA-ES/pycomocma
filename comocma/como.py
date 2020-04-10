#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the implementation of the Sofomore algorithm in 2 objectives, defined in 
the paper [Toure, Cheikh, et al. "Uncrowded Hypervolume Improvement: 
        COMO-CMA-ES and the Sofomore framework." 
        GECCO'19-Genetic and Evolutionary Computation Conference. 2019.].
"""

from __future__ import division, print_function, unicode_literals
__author__ = "Cheikh Toure and Nikolaus Hansen"
__license__ = "BSD 3-clause"
__version__ = "0.5.0"
del division, print_function, unicode_literals

import ast
import numpy as np
import cma
from cma import interfaces
from nondominatedarchive import NonDominatedList
from moarchiving import BiobjectiveNondominatedSortedList
import warnings
import cma.utilities.utils
import os
from sofomore_logger import SofomoreDataLogger
import random
#import sys

class IndicatorFront:
    """with `hypervolume_improvement` method based on a varying empirical front.
    
    The front is either all kernels but one or based on the
    `list_attribute` of `moes` (like `archive`) as given on
    initialization.
    
    Usage::

        >> moes.front_observed = IndicatorFront()
        >> # OR
        >> moes.front_observed = IndicatorFront(list_attribute='archive')
        >> [...]
        >>
        >> moes.front_observed.set_kernel(moes[3], moes)
        >>
        >> f_points = [moes.front_observed.hypervolume_improvement(point)
        ..             for point in points] 

    """
    def __init__(self, list_attribute=None, NDA=None):
        """``getattr(moes, list_attribute)`` contains the list to create the front.
        
        `NDA` is a non-dominated archive with a `hypervolume_improvement` method.

        """
        self.list_attribute = list_attribute
        self.NDA = NDA or BiobjectiveNondominatedSortedList
        self.kernel = None  # current active kernel
        self.front = None  # instance of NDA

    def hypervolume_improvement(self, point):
        return self.front.hypervolume_improvement(point)

    def set_kernel(self, kernel, moes, lazy=True):
        """Set empirical front for evolving the given kernel.
        
        By default, make changes only when kernel has changed.
        
        Details: ``moes.reference_point`` and, in case, its attribute
        with name `self.list_attribute: str` is used.
        """
        if lazy and kernel == self.kernel:
            return
        if self.list_attribute:
            self.front = self.NDA(getattr(moes, self.list_attribute),
                                  moes.reference_point)
        else:
            self.front = self.NDA([k.objective_values for k in moes if k != kernel],
                                  moes.reference_point)
        self.kernel = kernel

class Sofomore(interfaces.OOOptimizer):
    """ 
    Sofomore framework for multiobjective optimization, with the 
    ask-and-tell interface.
    See: [Toure, Cheikh, et al. "Uncrowded Hypervolume Improvement: 
        COMO-CMA-ES and the Sofomore framework." 
        GECCO'19-Genetic and Evolutionary Computation Conference. 2019.].
        
    Calling Sequences
    =================

    - ``moes = Sofomore(list_of_solvers_instances, opts, reference_point)``

    - ``moes = Sofomore(list_of_solvers_instances, reference_point)``

    - ``moes = Sofomore(list_of_solvers_instances, 
                             reference_point).optimize(objective_fcts)``

    Arguments   
    =========
    `list_of_solvers_instances`
        list of instances of single-objective solvers.
        It's generally created via a factory function.
        Let's take the example of a factory function that returns cma-es 
        instances, called `get_cmas`. Then:
        ``list_of_solvers_instances = get_cmas(11 * [x0], sigma0)`` 
        creates a list of 11 cma-es instances of initial mean `x0` and initial 
        step-size `sigma0`.
        A single-objective solver instance must have the following
        attributes and methods:
            - `incumbent`: an attribute that gives an estimate of 
            the single-objective solver.
            - `objective_values`: an attribute that stores the objective 
            values of the incumbent.
            - `stop`: a method returning a dictionary representing the 
            termination status.
            - `ask`: generates new candidate solutions.
            - `tell`: passes the objective values and updates the states of the
            single-objective solvers.
                
            
    `opts`
        opts, a dictionary with optional settings related to the 
        Sofomore framework. It contains the following keys:
            - 'archive': default value is `True`. 
            If its value is `True`, tracks the non-dominated
            points among the points evaluated so far during the 
            optimization.
            The archive will not interfere with the optimization 
            process.
            - 'update_order': default value is a function that takes a natural 
            integer as input and return a random number between 0 and 1.
            It is used as a `key value` in: `sorted(..., key = ...)`, and guides the
            order in which the kernels will be updated during the optimization.
 

    `reference_point`  
        reference point of the multiobjective optimization.
        Its default value is `None` but should be set by the user 
        beforehand to guarantee an optimal p-distribution convergence of 
        the Hypervolume indicator of `p` points towards the Pareto set/front.
        It can be changed dynamically by the user if needed.
        

    Main interface / usage
    ======================
    The interface is inherited from the generic `OOOptimizer`
    class, which is the same interface used by the python cma. An object 
    instance is generated as following:
        
        list_of_solvers_instances = como.get_cmas(11 * [x0], sigma0)
        moes = como.Sofomore(list_of_solvers_instances,
                             opts = opts,
                           reference_point = reference_point)

    The least verbose interface is via the optimize method::

         moes.optimize(objective_func)
         where `objective_func` is a callable multiobjective function

    More verbosely, the optimization of the multiobjective function 
    `objective_funcs` is done using the methods `stop`, `ask`, and `tell`::
        
        while not moes.stop():
            solutions = moes.ask()
            objective_values = [objective_funcs(x) for x in solutions]
            moes.tell(solutions, objective_values)
            moes.disp()

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
               list_of_solvers_instances, # usually come from a factory function 
                                         #  creating single solvers' instances
               opts = None, # keeping an archive, decide whether we use restart, etc.
               reference_point = None,    
               ):
        """
        Initialization:
            - `list_of_solvers_instances` is a list of single-objective 
            solvers' instances
            - `opts` is a dictionary updating the values of 
            `archive` and `update_order`, that responds respectfully to whether
            or not tracking an archive, and the order of update of the kernels.
            - The reference_point is set by the user during the 
            initialization.
        """
        assert len(list_of_solvers_instances) > 0
        self.kernels = list_of_solvers_instances
        self.num_kernels = len(self.kernels)
        self.dimension = self.kernels[0].N
        self._active_indices = list(range(self.num_kernels))

        for kernel in self.kernels:
            if not hasattr(kernel, 'objective_values'):
                kernel.objective_values = None
        self.reference_point = reference_point
        defopts = {'archive': True, 'restart': None, 'verb_filenameprefix': 'outsofomore' + os.sep, 
                   'verb_log': 1, 'verb_disp': 100, 'update_order': sort_random,
                   'continue_stopped_kernel': False, # when True and restarts=True, will continue stopped kernel
                   'random_restart_on_domination': False, # when True, do random restart if stopped kernel is dominated
                   'increase_popsize_on_domination': False,
                   'indicator_front': None  # 'archive' or any attribute containing a list of f-pairs
                   }
        if opts is None:
            opts = {}
        if isinstance(opts, dict):
            defopts.update(opts)
        else:
            warnings.warn("options should be either a dictionary or None.")
        self.opts = defopts
        self.restart = self.opts['restart']
        self.isarchive = self.opts['archive']
        if self.isarchive:
            self.archive = []
        self.NDA = None # the callable for nondominated archiving
        self.indicator_front = IndicatorFront(self.opts['indicator_front'])
        self.offspring = []
        self._told_indices = range(self.num_kernels)
        
        self.key_sort_indices = self.opts['update_order']
        self.countiter = 0
        self.countevals = 0
        self._remaining_indices_to_ask = range(self.num_kernels) # where we look when calling `ask`
        self.logger = SofomoreDataLogger(self.opts['verb_filenameprefix'],
                                                     modulo=self.opts['verb_log']).register(self)
        self.best_hypervolume_pareto_front = 0.0
        self.epsilon_hypervolume_pareto_front = 0.1 # the minimum positive convergence gap
        self._ratio_nondom_offspring_incumbent = self.num_kernels * [0]
        
        self._last_stopped_kernel_id = None
        self._number_of_calls_best_chv_restart = 0
        self._number_of_calls_random_restart = 0

        self.popsize_random_restart = self.kernels[0].popsize

    def __iter__(self):
        """
        make `self` iterable. 
        """
        return iter(self.kernels)
    
    def __getitem__(self, i):
        """
        make `self` subscriptable.
        """
        return self.kernels[i]
    
    def __len__(self):
        """return length of the `Sofomore` instance by calling ``len(.)``.
        
        The length is the number of (active and inactive) kernels
        and hence consistent with subscription like
        ``[moes[i] for i in range(len(moes)) if i in moes._active_indices]``.
        """
        return len(self.kernels)

    def _UHVI_indicator_archive(self, kernel):
        """return archive for uncrowded hypervolume improvement indicator for `kernel`.

        `kernel` can also be the respective index in `self`.
        """
        # TODO: checking the list of f_pairs for copies and 
        # looking at contributing_hypervolume is cheaper?
        try:  # allow kernel index as shortcut for kernel
            kernel = self[kernel]
        except TypeError:
            pass
        return self.NDA([k.objective_values for k in self
                            if k != kernel and k.objective_values is not None],
                        self.reference_point)

    def _UHVI_indicator(self, kernel):
        """return indicator function(!) for uncrowded hypervolume improvement for `kernel`.

        If `moes` is a `Sofomore` class instance::
    
            >> moes._UHVI_indicator(moes[1])(moes[2].objective_values)

            or::

            >> moes._UHVI_indicator(1)(moes[2].objective_values)

        both return the UHVI indicator function for kernel 1 and evaluate
        kernel 2 on it::

            >> [moes._UHVI_indicator(k)(k.objective_values)] for k in moes]

        is the list of UHVI values for all kernels where kernels occupying the
        very same objective value have indicator value zero.
        """
        return self._UHVI_indicator_archive(kernel).hypervolume_improvement

    def sorted(self, key=None, reverse=True, **kwargs):
        """return a reversed sorted list of kernels.

        By default kernels are reversed sorted by HV contribution or UHVI
        (which we aim to maximize) in the set of kernels. Exact copies have
        zero or negative UHVI value.

            >> moes.sorted(key = lambda k: moes.archive.contributing_hypervolume(k.objective_values))

        sorts w.r.t. archive contribution (clones may get positive contribution).

        """
        def hv_improvement(kernel):
            if kernel.objective_values is None:
                return float('-inf')
            return self._UHVI_indicator(kernel)(kernel.objective_values)
        if key is None:
            key = hv_improvement
        return sorted(self, key=key, reverse=reverse, **kwargs)

    def ask(self, number_to_ask = 1):
        """
        get the kernels' incumbents to be evaluated and sample new candidate solutions from 
        `number_to_ask` kernels.
        The sampling is done by calling the `ask` method of the
        `cma.CMAEvolutionStrategy` class.
        The indices of the considered kernels' incumbents are given by the 
        `_told_indices` attribute.
        
        To get the `number_to_ask` kernels, we use the function `self.key_sort_indices` as
        a key to sort `self._remaining_indices_to_ask` (which is the list of
        kernels' indices wherein we choose the first `number_to_ask` elements.
        And if `number_to_ask` is larger than `len(self._remaining_indices_to_ask)`,
        we select the list `self._remaining_indices_to_ask` extended with the  
        first `number_to_ask - len(self._remaining_indices_to_ask)` elements
        of `range(self.num_kernels)`, sorted with `self.key_sort_indices` as key.

        Arguments
        ---------
        - `number_to_ask`: the number of kernels where we sample solutions
         from, it's of type int and is smaller or equal to `self.num_kernels`
        
        Return
        ------
        The list of the kernels' incumbents to be evaluated, extended with a
        list of N-dimensional (N is the dimension of the search space) 
        candidate solutions generated from `number_to_ask` kernels 
        to be evaluated.
    
        :See: the `ask` method from the class `cma.CMAEvolutionStrategy`,
            in `evolution_strategy.py` from the `cma` module.
            
        """
        # TODO: be specific about the serial and the parallel case
        if number_to_ask == "all":
            number_to_ask = len(self._active_indices)
        assert number_to_ask > 0
        if number_to_ask > len(self._active_indices):
            number_to_ask = len(self._active_indices)
            warnings.warn("value larger than the number of active kernels {}. ".format(
                    len(self._active_indices)) + "Set to {}.".format(len(self._active_indices)))
        self.offspring = []
        res = [self.kernels[i].incumbent for i in self._told_indices]
        indices_to_ask = self._indices_to_ask(number_to_ask)
        for ikernel in indices_to_ask:
            kernel = self.kernels[ikernel]
            offspring = kernel.ask()
            res.extend(offspring)
            self.offspring += [(ikernel, offspring)]

        return res
        
    def tell(self, solutions, objective_values):
        """
        pass objective function values to update the state variables of some 
        kernels, `self._told_indices` and eventually `self.archive`.
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
        if self.NDA is None:
            self.NDA = BiobjectiveNondominatedSortedList if len(
                    objective_values[0]) == 2 else NonDominatedList
        
        objective_values = np.asarray(objective_values).tolist()

        for i in range(len(self._told_indices)):
            self.kernels[self._told_indices[i]].objective_values = objective_values[i]
        
        if self.reference_point is None:
            pass #write here the max among the kernel.objective_values       
            
        start = len(self._told_indices) # position of the first offspring
        self._told_indices = []
        for ikernel, offspring in self.offspring:
            self.indicator_front.set_kernel(self[ikernel], self)  # use reference_point and list_attribute
            hypervolume_improvements = [self.indicator_front.hypervolume_improvement(point)
                                            for point in objective_values[start:start+len(offspring)]]
            kernel = self.kernels[ikernel]
            if kernel.fit.median0 is not None and kernel.fit.median0 >= 0:
                # make sure the median reference comes from the right side of the empirical front
                # was: ikernel in self._active_indices and kernel.objective_values not in self.pareto_front:
                # a hack to prevent early termination of dominated kernels
                # from the `tolfunrel` condition.
                # TODO: clean implementation, proposal:
                #   if self.indicator_front.hypervolume_improvement(kernel.objective_values) <= 0:  # kernel.fit.median0 >= 0 is the same
                #       kernel.stop(reset='tolfunrel')  # to be implemented
                kernel.fit.median0 = None
            kernel.tell(offspring, [-float(u) for u in hypervolume_improvements])
            
            # investigate whether `kernel` hits its stopping criteria
            if kernel.stop():
                self._active_indices.remove(ikernel) # ikernel must be in `_active_indices`
                self._last_stopped_kernel_id = ikernel
                # update of self.popsize_random_restart
                if self.opts['increase_popsize_on_domination']:
                    popsize_dominated_kernels = [k.popsize for k in self.kernels
                                                 if k is not kernel and  # we can't yet say whether kernel is dominated
                                                     k.objective_values not in self.pareto_front]
                    #print(len(popsize_dominated_kernels), "dominated kernels")
                    if popsize_dominated_kernels and max(popsize_dominated_kernels) == self.popsize_random_restart:
                        self.popsize_random_restart *= 2
                if self.restart is not None:
                    kernel_to_add = self.restart(self)
                    self._told_indices += [self.num_kernels]
                    self.add(kernel_to_add)

            try:
                kernel.logger.add()
            except:
                pass
            kernel._last_offspring_f_values = objective_values[start:start+len(offspring)]
            
            start += len(offspring)
            
        self._told_indices += [u for (u,v) in self.offspring]
        
        current_hypervolume = self.pareto_front.hypervolume
        epsilon = abs(current_hypervolume - self.best_hypervolume_pareto_front)
        if epsilon:
            self.epsilon_hypervolume_pareto_front = min(self.epsilon_hypervolume_pareto_front, 
                                                        epsilon)
        self.best_hypervolume_pareto_front = max(self.best_hypervolume_pareto_front,
                                                 current_hypervolume)

        if self.isarchive:
            if not self.archive:
                self.archive = self.NDA(objective_values, self.reference_point)
            else:
                self.archive.add_list(objective_values)
        self.countiter += 1
        self.countevals += len(objective_values)

        
    @property
    def pareto_front(self):
        """
        return the non-dominated solutions among the kernels'
        objective values.
        It's the image of `self.pareto_set`.
        """
        return self.NDA([kernel.objective_values for kernel in self.kernels \
                         if kernel.objective_values is not None],
                         self.reference_point)

    @property
    def pareto_set(self):
        """
        return the non-dominated solutions among the kernels'
        incumbents.
        It's the pre-image of `self.pareto_front`.
        """
        return [kernel.incumbent for kernel in self.kernels if \
                kernel.objective_values in self.pareto_front]

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
        # update `_active_indices` from scratch: inactive kernels might be added
        self._active_indices = [idx for idx in range(self.num_kernels) if \
                                not self.kernels[idx].stop()]
        self._ratio_nondom_offspring_incumbent = self.num_kernels * [0] # self.num_kernels changed
        
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
        # update `_active_indices`
        self._active_indices = [idx for idx in range(self.num_kernels) if \
                                not self.kernels[idx].stop()]

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
    
    def _indices_to_ask(self, number_to_ask):
        """
        """
        sorted_indices = sorted(self._remaining_indices_to_ask, key = self.key_sort_indices)
        indices_to_ask = []
        remaining_indices = []
        if number_to_ask <= len(sorted_indices):
            indices_to_ask = sorted_indices[:number_to_ask]
            remaining_indices = sorted_indices[number_to_ask:]
        else:
            val = number_to_ask - len(sorted_indices)
            indices_to_ask = sorted_indices
            sorted_indices = sorted(self._active_indices, key = self.key_sort_indices)
            indices_to_ask += sorted_indices[:val]
            remaining_indices = sorted_indices[val:]
        
        self._remaining_indices_to_ask = remaining_indices
        return indices_to_ask
    
    def inactivate(self, kernel):
        """
        inactivate `kernel`, assuming that it's an element of `self.kernels`,
        or an index in `range(self.num_kernels)`.
        When inactivated, `kernel` is no longer updated, it is ignored.
        However we do not remove it from `self.kernels`, meaning that `kernel`
        might still play a role, due to its eventual trace in `self.pareto_front`.
    
        """
        if kernel in self.kernels:
            ikernel = self.kernels.index(kernel)

        try:
            self._active_indices.remove(ikernel)
            self.kernels[ikernel].opts['termination_callback'] += (lambda _: 'kernel turned off',)
        except (AttributeError, TypeError, KeyError, ValueError):
            warnings.warn("check again if `opts['termination_callback']` is"+
                          " correctly used, or if the kernel is not already"+
                          " turned off.")
            
    def activate(self, kernel):
        """
        activate `kernel` when it was inactivated beforehand. Otherwise 
        it remains quiet.
        
        We expect the kernel's `stop` method in interest to look like:
        kernel.stop() = {'callback': ['kernel turned off']}
        """
        raise NotImplementedError
        if kernel in self.kernels:
            ikernel = self.kernels.index(kernel)
        new_list = [callback for callback in self.kernels[ikernel].opts['termination_callback']\
                if callback(kernel) == 'kernel turned off']
        kernel.opts['termination_callback'] = new_list
        if not kernel.stop():
            self._active_indices += [ikernel]

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
    
def get_kernel_random_restart(moes, x0_fct=None, opts=(), tolx_factor=0.05, **kwargs):
    """return a `list` with one element of type `CmaKernel`.
    
    Parameters
    ----------
    moes: `Sofomore`, the instance for which this method is used as
    `restart` option.

    x0_fct: `callable`, that returns an initial solution passed to
    `CmaKernel`.

    opts: `dict`, options passed (possibly with modifications) to
    `CmaKernel(cma.CMAEvolutionStrategy)`.

    kwargs: `dict`, unused keyword arguments to allow for a generic call of
    `Sofomore.restart`.

    Details: this function mimics `random_restart_kernel` but uses
    additionally `cma_kernel_default_options_dynamic_tolx` to control the
    `tolx` option of `CmaKernel`.
"""

    def x0(moes):
        """x0 in [-5, 5]"""
        return 10 * np.random.rand(moes.dimension) - 5
    def sigma0(moes):
        """sigma0 from first kernel"""
        return moes[0].sigma0
    def get_opts(moes):
        """inopts from last stopped kernel (for backwards "compatiblity")"""
        if moes._last_stopped_kernel_id is not None:
            opts_ = dict(moes[moes._last_stopped_kernel_id].inopts or ())
        else:  # fall back to inopts of first kernel
            opts_ = dict(moes[0].inopts or ())
        opts_['tolx'] = cma_kernel_default_options_dynamic_tolx(
                            moes, factor=tolx_factor)
        opts_.update(opts or ())  # catch None
        return opts_
    res = get_cmas((x0_fct or x0)(moes), sigma0(moes),
                         get_opts(moes), len(moes))
    res[0]._rampup_method = get_kernel_random_restart
    return res

def get_kernel_best_chv_restart(moes, opts=(), **kwargs):
    """return a `list` with one element of type `CmaKernel`.
    
    Parameters
    ----------
    moes: `Sofomore`, the instance for which this method is used as
    `restart` option.

    opts: `dict`, options passed (possibly with modifications) to
    `CmaKernel(cma.CMAEvolutionStrategy)`.

    kwargs: `dict`, unused keyword arguments to allow for a generic call of
    `Sofomore.restart`.

    Details: this function picks best boundary kernels only with
    probability of about 2 / number_of_kernels and uses
    `cma_kernel_default_options_dynamic_tolx` to control the `tolx` option
    of `CmaKernel`.

    TODO: it may be better to pick best boundary kernels with at least,
    say, 5%. However from a practical perspective, boundary kernels are
    usually of lesser interest and pushing the boundary leads to more gaps
    at extremer (less interesting) regions(?) that will also be filled in
    the sequel.
"""

    def pick_kernel(moes):
        for k in moes.sorted():
            if k.objective_values in [moes.pareto_front[i]
                                      for i in [0, -1]]:
                if len(moes) < 3 or np.random.randint(len(moes)) < 2:
                    return k
            else:
                break
        return k
    def get_opts(moes):
        opts_ = {'tolx': cma_kernel_default_options_dynamic_tolx(moes, factor=0.05)}
        opts_.update(opts or ())  # or () catches opts == None
        opts_.update([['verb_filenameprefix', # currently unavoidable code duplication from line ~1600 of get_cmas
                      os.path.join('cma_kernels', str(len(moes)))]])
        return opts_
    kernel = pick_kernel(moes)
    res = kernel._copy_light(sigma=kernel.sigma0, inopts=get_opts(moes))
    res._rampup_method = get_kernel_best_chv_restart
    return [res]

def random_restart_kernel(moes, x0_fct=None, sigma0=None, opts=None, **kwargs):
    
    """create a kernel (solver) of TYPE CmaKernel with a random initial mean, or 
    an initial mean given by the factory function `x0_funct`, and initial step-size
    `sigma0`.
    
    Parameters
    ----------
    moes : TYPE Sofomore
        A multiobjective solver instance with cma-es (of TYPE CmaKernel) solvers.
    x0_fct : TYPE function, optional
        A factory function that creates an initial mean. The default is None.
    sigma0 : TYPE float, optional
        Initial step-size of the returned kernel. The default is None.
    opts : TYPE dict, optional
        The returned kernel's options. The default is `None`.
    **kwargs : 
        Other keyword arguments.

    Returns
    -------
    A kernel (solver) of TYPE CmaKernel.

    """

    if x0_fct is not None:
        x0 = x0_fct(moes.dimension)  # or however we can access the current search space dimension
    else:
        x0 = 10 * np.random.rand(moes.dimension) - 5  # TODO: or whatever we want as default
    if sigma0 is None: 
        sigma0 = moes[0].sigma0 / 1.  # decrease the initial  step-size ?
    
    my_opts = {}  # we use inopts to avoid copying the initialized random seed
    for op in (moes[moes._last_stopped_kernel_id].inopts, opts):
        if op is not None:
            my_opts.update(op)
            

    my_opts.update({'popsize': moes.popsize_random_restart})
    
    return  get_cmas(x0, sigma0, my_opts, moes.num_kernels)
    
def best_chv_restart_kernel(moes, sigma_factor=1, **kwargs):
    """create a kernel (solver) of TYPE CmaKernel by duplicating the kernel with 
    best uncrowded hypervolume improvement.
    
    Parameters
    ----------
    moes : TYPE Sofomore
        A multiobjective solver instance with cma-es (of TYPE CmaKernel) solvers.
    sigma_factor : TYPE int or float, optional
        A step size factor used in the initial step-size of the kernel returned.
        The default is 1.
    **kwargs : 
        Other keyword arguments.

    Returns
    -------
    A kernel (solver) of TYPE CmaKernel derived from the kernel with largest contributing HV.

    """

    # test if stopped kernel is truly dominated by another kernel
    # if yes, do an independent random restart

    if moes.opts['random_restart_on_domination']:
        kernel = moes.kernels[moes._last_stopped_kernel_id]
        if kernel.objective_values not in moes.pareto_front:
            return random_restart_kernel(moes)

    hvc = []
    for idx in range(moes.num_kernels):
        front = moes.NDA([moes.kernels[i].objective_values for i in range(moes.num_kernels) if i != idx],
                            moes.reference_point)
        f_pair = moes.kernels[idx].objective_values
        hvc += [front.hypervolume_improvement(f_pair)]
    sorted_indices = sorted(range(moes.num_kernels), key=lambda i: - hvc[i])
    my_front = moes.pareto_front
    idx_best = sorted_indices[0]
    if len(my_front) > 1:
        for i in sorted_indices:
            kernel = moes.kernels[i]
            if kernel.stop() or kernel.objective_values not in [my_front[0], my_front[-1]]:
                idx_best = i
                break
    ker = moes.kernels[idx_best]
    new_sigma0 = sigma_factor * ker.sigma
    if moes.opts['continue_stopped_kernel'] and idx_best not in moes._active_indices:
        moes._active_indices.append(idx_best)
        ker.sigma = new_sigma0

    newkernel = ker._copy_light(sigma=new_sigma0, inopts={'verb_filenameprefix': 'cma_kernels' + os.sep +
                                                                     str(moes.num_kernels)})
    return [newkernel]

def _old_best_chv_or_random_restart_kernel_old(moes, sigma_factor=1, x0_fct=None, sigma0=None, opts=None, **kwargs):
    """DEPRECATED, use instead ``como.RampUpSelector([como.random_restart_kernel, como.best_chv_restart_kernel])``
    and `functools.partial` to assign parameters to the functions beforehand.

    generate fairly, via a derandomized scenario, either a kernel via `best_chv_restart_kernel`
    or a kernel via `random_restart_kernel`.
    
    Parameters
    ----------
    moes : TYPE Sofomore
        A multiobjective solver instance with cma-es solvers.
    sigma_factor : TYPE int or float, optional
        A step size factor used in the initial step-size of the kernel created via
        the function `best_chv_restart_kernel`. The default is 1.
    x0_fct : TYPE function, optional
        A factory function that creates an initial mean for the factory function
        `random_restart_kernel`. The default is None.
    sigma0 : TYPE float, optional
        Initial step-size of a kernel created via `random_restart_kernel`.
        The default is None.
    opts : TYPE dict, optional
        The created kernel's options. The default is {}.
    **kwargs : 
        Other keyword arguments.

    Returns
    -------
    A solver of TYPE CmaKernel.

    DEPRECATED.
    """
    
    assert abs(moes._number_of_calls_best_chv_restart - moes._number_of_calls_random_restart) < 2
    
    if moes._number_of_calls_best_chv_restart < moes._number_of_calls_random_restart:
        moes._number_of_calls_best_chv_restart += 1
        assert moes._number_of_calls_best_chv_restart == moes._number_of_calls_random_restart
        return best_chv_restart_kernel(moes, sigma_factor, **kwargs)
    
    if opts is None:
        opts = {}
        
    if moes._number_of_calls_best_chv_restart > moes._number_of_calls_random_restart:
        moes._number_of_calls_random_restart += 1
        assert moes._number_of_calls_best_chv_restart == moes._number_of_calls_random_restart
        return random_restart_kernel(moes, x0_fct, sigma0, opts, **kwargs)
    
    p = random.random()
    if p < 0.5:
        moes._number_of_calls_best_chv_restart += 1
        return best_chv_restart_kernel(moes, sigma_factor, **kwargs)
    
    moes._number_of_calls_random_restart += 1
    return random_restart_kernel(moes, x0_fct, sigma0, opts, **kwargs)

class _CounterDict(dict):
    """A dictionary with two additional features.
    
    1) it can be initialized by keywords only, setting all values to 0.
    
    2) the `argmin` method gives the key associated to the smallest value,
    breaking ties at random.

    `_CounterDict` is somewhat a misnomer, as any type than can be
    sorted can be used as values.

    >>> from comocma.como import _CounterDict
    >>> keys = [1, 2]
    >>> bs = _CounterDict(keys)
    >>> assert bs == _CounterDict(zip(keys, len(keys) * [0]))
    >>> assert bs[bs.argmin()] == min(bs.values())
    >>> bs[2] = -3
    >>> assert bs.argmin() == 2

    """
    def __init__(self, *args, **kwargs):
        try:
            super(_CounterDict, self).__init__(*args, **kwargs)
        except TypeError:  # make values to be zero
            super(_CounterDict, self).__init__(zip(args[0], len(args[0]) * [0]))
        self.argmin()  # assign `last` attribute
        self.last = self.last  # declare `last` attribute
        """  all minimal keys on last call of `argmin`"""
    def argmin(self):
        m = min(self.values())
        self.last = [k for k in self if self[k] == m]
        return self.last[np.random.randint(len(self.last))]

class RampUpSelector:
    """Takes a list of ramp-up methods and a selection "criterion",
    
    on inialization. When called, calls the ramp-up method with the
    smallest cumulative criterion value and returns the result.
    
    The default selection criterion is number of ramp-up calls. Otherwise,
    `criterion` can be an attribute name or a `callable`. A string
    signifies an attribute name of the returned result of the ramp-up
    methods. A `callable` is called with the result as argument.

    For example, ``criterion='countevals'`` or ``lambda k: k.countevals``
    uses the sum of ``result.countevals`` (evaluated only right before the
    next ramp up) as criterion which method to use next.

    The `costs` attribute stores the sum of selection criterion values
    for each method in a dictionary.
    
    Usage:

    >>> from comocma import como
    >>> restart_methods = (como.get_kernel_random_restart,
    ...                    como.get_kernel_best_chv_restart)
    >>> selected_restarts = como.RampUpSelector(restart_methods)

    or:

    >>> selected_restarts = como.RampUpSelector(restart_methods,
    ...                                         criterion='countevals')

    thereby assuming that the return value of the restart methods has the
    attribute `countevals` to be used to sum up the costs. Now calling
    `selected_restarts` has the same interface as either of the
    `restart_methods`, that is, the same calling arguments and the same
    return value(s), and it can be used just like the original single
    methods::

        >> moes = como.Sofomore(list_of_solvers,
        ..                      {'restart': selected_restarts})

    as restart option.
"""
    def __init__(self, rampup_methods, criterion=None):
        self.rampup_methods = rampup_methods
        self.criterion = criterion
        """  a callable or an attribute name, may be changed any time"""
        self.costs = _CounterDict(self.rampup_methods)
        self.counts = _CounterDict(self.rampup_methods)
        self.method = None
        """ last used rampup method"""
        self.result = None
        """ last rampup result, this may or may not be a list"""

    def _update_costs(self):
        """add cost value of finished (last ramped) method"""
        if not self.method:
            return  # do nothing before to know which method was called
        if self.criterion is None:
            val = 1
        else:
            try:  # a string leads to attribute access
                # this is a hack and should not be necessary:
                if isinstance(self.result, (list, tuple)): 
                    val = sum(getattr(k, self.criterion) for k in self.result)
                else:  # should not happen anymore within `como`
                    val = getattr(self.result, self.criterion)
            except TypeError:  # an attribute must be a string
                val = self.criterion(self.result)
        self.costs[self.method] += val
        self.counts[self.method] += 1

    def __call__(self, *args, **kwargs):
        """update, select, and ramp up"""
        self._update_costs()  # depends on the final state of the previously returned result
        self.method = self.costs.argmin()
        self.result = self.method(*args, **kwargs)
        # if need be we could hack here to tweak result further before to deliver
        return self.result


# callbacks for sorting indices to pick in the `tell` method.
# This is useful in the case where num_to_ask is equal to 1,
# so that the kernels are updated in the `tell` method one by one
        
def sort_even_odds(i):
    """
    pick the kernels with even indices before the kernels with odd indices in
    the `tell` method
    """
    return i % 2

def sort_odds_even(i):
    """
    pick the kernels with odd indices before the kernels with even indices in
    the `tell`method
    """
    return - (i % 2)

def sort_random(i):
    """
    randomly pick the kernels to update in the `tell` method
    """
    return np.random.rand()

def sort_increasing(i):
    """
    update respectively `self.kernels[0]`, `self.kernels[1]`, ..., `self.kernels[-1]`
    """
    return i

def sort_decreasing(i):
    """
    update respectively `self.kernels[-1]`, `self.kernels[-2]`, ..., `self.kernels[0]`
    """
    return - i

cma_kernel_default_options_replacements = {
        'conditioncov_alleviate': [np.inf, np.inf],
        'verbose': -1,
        'tolx': 1e-4,
        'tolfunrel': 0,
        'tolfun': 0,
        'tolfunhist': 0,
        'tolstagnation': 1e23,  # should become float('inf'),
        }

def cma_kernel_default_options_dynamic_tolx(moes, factor=0.1):
    """return `factor` times minimum `tolx` from non-dominated kernels.

    Fallback to default `tolx` if the `pareto_front` is empty.
"""
    if moes.pareto_front:
        return factor * min(k.stop(get_value='tolx') for k in moes
                            if k.objective_values in moes.pareto_front)
    if 'tolx' in cma_kernel_default_options_replacements:
        return cma_kernel_default_options_replacements['tolx']
    return cma.CMAOptions().eval('tolx')

### Factory function to create cma-es:

def get_cmas(x_starts, sigma_starts, inopts=None, number_created_kernels=0):
    """
    Factory function that produces `len(x_starts)` instances of type `cmaKernel`.
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
            mybounds = list_of_opts[i]['bounds']
            if isinstance(mybounds, str):
                mybounds = ast.literal_eval(mybounds)
            bounds_transform = cma.constraints_handler.BoundTransform(mybounds)
            x_starts[i] = bounds_transform.repair(x_starts[i])
        except KeyError:
            pass
    
    for i in range(num_kernels):
        defopts = cma.CMAOptions()
        defopts.update(cma_kernel_default_options_replacements)        
        if isinstance(list_of_opts[i], dict):
            defopts.update(list_of_opts[i])
        defopts.update({'verb_filenameprefix': 'cma_kernels' + os.sep + 
                        str(number_created_kernels+i)})
            
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
        self._last_offspring_f_values = None # the fvalues of its offspring
        # used in the last call of `tell`.  
    
    @property
    def incumbent(self):
        """
        it gives the 'repaired' mean of a cma-es. For a problem with bound
        constraints, `self.incumbent` in inside the bounds.
        """
        return self.boundary_handler.repair(self.mean)
    
    def _copy_light(self, sigma=None, inopts=None):
        """tentative copy of self, versatile (interface and functionalities may change).
        
        This may not work depending on the used sampler.
        
        Copy mean and sample distribution parameters and input options.

        Do not copy evolution paths, termination status or other state variables.
        """
        es = super(CmaKernel, self)._copy_light(sigma, inopts)

        es.objective_values = self.objective_values
        es._last_offspring_f_values = self._last_offspring_f_values
        return es  
    
class FitFun:
    """
    Define a callable multiobjective function from single objective ones.
    Example:
        fitness = como.FitFun(cma.ff.sphere, lambda x: cma.ff.sphere(x-1)).
    """
    def __init__(self, *args):
        self.callables = args
    def __call__(self, x):
        return [f(x) for f in self.callables]


