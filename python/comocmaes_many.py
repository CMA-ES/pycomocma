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
               #  list_of_x0,
               #  sigma0,        # related to como
               #  factory_funciton,# put a kernel creating function here
               list_of_solver_instances,  
               options, # we put the order here: update_order = lambda x: np.arange(x),
                # solver_options = None, : should be in the factory function that creates kernels
                 reference_point = None,    
                 ):
        
        """
        """
       # self.dim = len(list_of_x0[0])
        self.num_kernels = len(list_of_solver_instances)
        self.sigma0 = sigma0
        self.reference_point = reference_point
        
        # the following should be inside the SO dolvers
        kernels = []
        for i in range(num_kernels):
            defopts = {'conditioncov_alleviate':[np.inf, np.inf],
                       'CMA_const_trace': 'False'}
            defopts.update(opts)
            defopts.update({'verb_filenameprefix' : str(
                    i)})
        
            
            kernels += [cma.CMAEvolutionStrategy(list_of_x0[i],sigma0,defopts)]#,'verbose':-9})]#,'AdaptSigma':cma.sigma_adaptation.CMAAdaptSigmaTPA})]
        self.kernels = kernels
        #Here we store the objective values of the cma means 
        #once and for all, without creating a new structure. 
        #Note that the fit class is currently "blanc":
        self.evaluate = fitness
        mean_values = self.evaluate(x0)
        for kernel in self.kernels:
            kernel.fit.fitness = mean_values
        
        if self.reference_piont == None:
         #   self.reference_point = [max([kernel.fit.fitness for kernel in self.kernels])]
           
        self.update_order = update_order
        num_objectives = len(mean_values)
        assert num_objectives > 1
        NDA = BNDS if num_objectives == 2 else NDL
        self.layer = NDA([kernel.fit.fitness for kernel in self.kernels],
                         reference_point)
     
        
    
    def ask(self, num_kernels = 1):
        """
        
        return a list of vectors
        """
        if num_kernels == "all":
            num_kernels = len(self.kernels)
        self.offspring = []
        res = []
        generator = self._randint_derandomized_generator(len(self.kernels), size=num_kernels)
        for ikernel in generator:
            kernel = self.kernels[ikernel]
            offspring = kernel.ask()
            res.extend([kernel.mean]+offspring)
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
            kernel.fit.fitness = self.evaluate(kernel.mean) # evaluate outside the 
            self.layer.add(kernel.fit.fitness)
            try:
                kernel.logger.add()
            except:
                pass
        
    def stop(self, tolx = None):
        """
        """
        if tolx == None:
            tolx = [self.kernels[i].opts['tolx'] for i in range(self.num_kernels)]
        assert all([u > 0 for u in tolx])
        if (all([all([self.kernels[i].sigma * xi < tolx[i] for xi
        in self.kernels[i].sigma_vec * self.kernels[i].pc]) 
        and all([self.kernels[i].sigma * xi < tolx[i] for xi in 
        self.kernels[i].sigma_vec * np.sqrt(self.kernels[i].dC)])
        for i in range(self.num_kernels)])):
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
