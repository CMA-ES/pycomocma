#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import cma
from cma import interfaces
from nondominatedarchive import NonDominatedList as NDL
from moarchiving import BiobjectiveNondominatedSortedList as BNDS

class Comocmaes(interfaces.OOOptimizer):
    
    """ 
    
    """
    def __init__(self,
                 x0,
                 sigma0,
                 evaluate,
                 num_kernels,
                 reference_point = None,    
                 update_order = lambda x: np.arange(x),
                 ):
        
        """
        """
        self.dim = len(x0)
        self.num_kernels = num_kernels
        self.sigma0 = sigma0
        self.reference_point = reference_point
      
        kernels = []
        for i in range(num_kernels):
            kernels += [cma.CMAEvolutionStrategy(x0,sigma0,{'verb_filenameprefix' : str(
                    i),'conditioncov_alleviate':[np.inf, np.inf],
'CMA_const_trace': 'False'})]#,'verbose':-9})]#,'AdaptSigma':cma.sigma_adaptation.CMAAdaptSigmaTPA})]
        self.kernels = kernels
        #Here we store the objective values of the cma means 
        #once and for all, without creating a new structure. 
        #Note that the fit class is currently "blanc":
        self.evaluate = evaluate
        mean_values = self.evaluate(x0)
        for kernel in self.kernels:
            kernel.fit.fitness = mean_values
           
        self.update_order = update_order
        num_objectives = len(mean_values)
        assert num_objectives > 1
        NDA = BNDS if num_objectives == 2 else NDL
        self.layer = NDA([kernel.fit.fitness for kernel in self.kernels],
                         reference_point)
     
        
    
    def ask(self, kernels = "all"):
        """
        """
        if kernels == "all":
            kernels = len(self.kernels)
        self.offspring = []
        res = []
        generator = self._randint_derandomized_generator(len(self.kernels), size=kernels)
        for ikernel in generator:
            kernel = self.kernels[ikernel]
            offspring = kernel.ask()
            res.extend(offspring)
            self.offspring += [(ikernel, offspring)]
        return res
        
    def tell(self, X, F):
        """
        """
        all_offspring = []
        solutions = [v for (u, v) in self.offspring]
        for v in solutions:
            all_offspring += v
        assert X == all_offspring
        
        start = 0 # position of the offspring
        for ikernel, offspring in self.offspring:
            kernel = self.kernels[ikernel]
            fit = kernel.fit.fitness
            if fit in self.layer:
                self.layer.remove(fit)
            hypervolume_improvements = [self.layer.hypervolume_improvement(
                    point) for point in F[start:start+len(offspring)]]
            start += len(offspring)
            kernel.tell(offspring, [-float(u) for u in hypervolume_improvements])
            kernel.fit.fitness = self.evaluate(kernel.mean)
            self.layer.add(kernel.fit.fitness)
            kernel.logger.add()
        
    def stop(self):
        """
        """
        if all(kernel.sigma < 10**-6 for kernel in self.kernels):
            return True
        return False
    
    def _randint_derandomized_generator(self, low, high=None, size=None):
        """the generator for `randint_derandomized`"""
        if high is None:
            low, high = 0, low
        if size is None:
            size = high
        delivered = 0
        while delivered < size:
            for randi in self.update_order(high - low):
                delivered += 1
                yield low + randi
                if delivered >= size:
                    break
