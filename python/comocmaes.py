#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Cheikh and ...
"""


import warnings
import fractions
import numpy as np
import cma
from moarchiving import BiobjectiveNondominatedSortedList as NDA
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from math import inf

class CoMoCmaes(object):
    
    """ 
    
    """
    def __init__(self,
                 objective_functions,    
                 dim,
                 sigma0,
                 num_kernels,
                 reference_point = None,    
                 max_evaluations = np.inf,
                 update_order = lambda x: np.arange(x),
                 kernels = None,
                 num_offsprings = None,
                 inner_iterations = 1
                 ):
        
        """
        """
        self.objective_functions = objective_functions
        self.dim = dim
        self.num_kernels = num_kernels
        self.sigma0 = sigma0
        self.num_offsprings = num_offsprings
        
        if not kernels:            
            kernels = []
            for i in range(num_kernels):
                x0 = np.random.rand(dim) * 0
                x0[0] = 0.5
                kernels += [cma.CMAEvolutionStrategy(x0,sigma0,{'verb_filenameprefix' : str(i)})]
        
        self.kernels = kernels
        self.max_evaluations = max_evaluations

        #Here we try a way to store the objective values of the cma means 
        #once and for all, without creating a new structure. 
        #Note that the fit class is currently "blanc":
        for kernel in self.kernels:
            kernel.fit.fitnesses = self.evaluate(kernel.mean)
        if not self.num_offsprings:
            self.num_offsprings = self.kernels[0].popsize
            
        self.inner_iterations = inner_iterations
        self.update_order = update_order
              
        self.layer = NDA([kernel.fit.fitnesses for kernel in self.kernels],
                        reference_point)
        self.hv = []
    
    def evaluate(self,x_var):
        """
        """
        return [ fun(x_var) for fun in self.objective_functions ]
    
        
    def step(self):
        
        """
        """
        
        order = self.update_order(len(self.kernels))
        for idx in range(len(self.kernels)):
            kernel = self.kernels[order[idx]]
            fit = kernel.fit.fitnesses
            if fit in self.layer:
                self.layer.remove(fit)
            for _ in range(self.inner_iterations):
                offsprings = kernel.ask()

                hypervolume_improvements = [self.layer.hypervolume_improvement(
                        self.evaluate(offspring)) for offspring in offsprings]

                kernel.tell(offsprings, [-float(u) for u in hypervolume_improvements])
                kernel.logger.add()    
    
    #updating the fitness:
            kernel.fit.fitnesses = self.evaluate(kernel.mean)
            self.layer.add(kernel.fit.fitnesses)
        self.hv += [self.layer.hypervolume]

                
    def run(self):
        """
        """
        #maxiter is the number of iterations based on the maximum budget which is max_evaluations

        maxiter = (self.max_evaluations-self.num_kernels)//(self.num_kernels*self.inner_iterations*(self.num_offsprings+1))
       # maxiter = 1

        for l in range(maxiter):
            self.step()
            if not (l % (maxiter//10)):
                print("{}".format(l/maxiter), end = ' ')

        for kernel in self.kernels:
            kernel.logger.plot()
            
if __name__ == "__main__":
    
    dim = 10
    num_kernels = 9
    b = np.zeros(dim)
    b[1] = 1
    fun = lambda x: cma.ff.cigtab(x)/cma.ff.cigtab(b),lambda x: cma.ff.cigtab(x-b)/cma.ff.cigtab(b)
    sigma0 = 0.5
    refpoint = [1.1, 1.1]
    max_evaluations = 7*10**4
    for __ in range(1):    
        mymo = CoMoCmaes(fun,dim,sigma0,num_kernels,refpoint,max_evaluations,
                       lambda x: np.random.permutation(x),inner_iterations = 1)
        mymo.run()
        plt.figure(18472)
        f1 = np.array([fun[0](mycma.mean) for mycma in mymo.kernels])
        f2 = np.array([fun[1](mycma.mean) for mycma in mymo.kernels])
        plt.plot(f1,f2,'o')
        plt.xlabel('first objective function')
        plt.ylabel('second objective function')
        plt.axes().set_aspect('equal')
        plt.title("Pareto front")
                    
        plt.figure(417283)
        maxiter = (mymo.max_evaluations-mymo.num_kernels)//(mymo.num_kernels*mymo.inner_iterations*(mymo.num_offsprings+1))
        myaxis = np.linspace(mymo.num_kernels*(
                mymo.kernels[0].popsize+1),maxiter*mymo.num_kernels*(mymo.kernels[0].popsize+1),maxiter)
        plt.semilogy(myaxis,[float(max(mymo.hv))-float(u) for u in mymo.hv],'-')
        plt.xlabel('function evaluations')
        plt.ylabel('log scale hypervolume gap with the Como algorithm')
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
 
