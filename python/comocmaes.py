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
from problems import BiobjectiveConvexQuadraticProblem as problem
import random


class CoMoCmaes(object):

    """

    """

    def __init__(self,
                 objective_functions,
                 dim,
                 sigma0, 
                 lbounds,
                 rbounds,
                 num_kernels,
                 reference_point=None,
                 max_evaluations=np.inf,
                 update_order=lambda x: np.arange(x),
                 kernels=None,
                 num_offspring=None,
                 inner_iterations=1,
                 lazy=False,
                 name=None
                 ):
        """
        Parameters :
        ==========
        - 'tuple of functions' objective_functions
        - 'positive integer' dim : dimension
        - 'positive float' sigma0 : initial stepsize for all the kernels
        - 'np.array(self.dim) float' lbounds, rbounds : left and right bounds
        between which the initial means of the kernels are picked randomly with
        uniform distribution
        - 'positive integer' num_kernels : number of kernels
        - 'list of floats of size len(objective_functions)' reference_point
        - 'positive integer' max_evaluations
        - 'fun' update_order : gives the update order of the kernels
        - 'list of num_kernels CMA-ES' kernels : initialization of the kernels
        - 'positive integer' num_offspring : population size of the single
        CMAES
        - 'positive integer' inner_iterations : number of iterations of each
        CMAES in one step
        - 'bool' lazy : if true, the kernel objective value is not removed
        during its update
        - 'string' name : the name of the objective function
        """
        self.objective_functions = objective_functions
        self.dim = dim
        self.num_kernels = num_kernels
        self.sigma0 = sigma0
        self.reference_point = reference_point
        self.counteval = 0
        self.lazy = lazy
        # initialization of kernels
        if kernels is None:
            kernels = []
            for i in range(num_kernels):
                x0 = lbounds + np.random.rand(self.dim)*(rbounds-lbounds)
                kernels += [cma.CMAEvolutionStrategy(x0, sigma0, {'verb_filenameprefix': str(
                    i), 'conditioncov_alleviate': [np.inf, np.inf],
                    'CMA_const_trace': 'True'})]  # ,'verbose':-9})]#,'AdaptSigma':cma.sigma_adaptation.CMAAdaptSigmaTPA})]
        self.kernels = kernels
        # initialization of num_offspring
        if num_offspring is None:
            num_offspring = self.kernels[0].popsize
        self.num_offspring = num_offspring
        self.max_evaluations = max_evaluations
        self.inner_iterations = inner_iterations
        self.update_order = update_order
        self.name = name

        # Here we try a way to store the objective values of the cma means
        # once and for all, without creating a new structure.
        # Note that the fit class is currently "blank":
        for kernel in self.kernels:
            # store the bi-objective values of each kernel's mean
            kernel.fit.fitnesses = self.evaluate(kernel.mean)
            # storing the ratio of non-dominated points among a kernel + its offspring:
            kernel.ratio_nondominated_offspring = []

        # self.layer is the list of the objective values of the non-dominated kernel means 
        # by non-dominated, we mean non-dominated by other current kernels means
        self.layer = NDA([kernel.fit.fitnesses for kernel in self.kernels],
                         reference_point)
        # self.archive is the list of the non-dominated objective values among
        # everything evaluated
        self.archive = NDA([kernel.fit.fitnesses for kernel in self.kernels],
                           reference_point)

        # storage attributes
        self.hv = []
        self.hv_archive = []
        self.ratio_nondominated_kernels = []
        self.ratio_nondominated_first_quartile_offspring = []
        self.ratio_nondominated_median_offspring = []
        self.ratio_nondominated_third_quartile_offspring = []

    def evaluate(self, x_var):
        """Evaluate the objective function in x_var and increment the number of
        evaluations."""
        self.counteval += 1
        return [fun(x_var) for fun in self.objective_functions]

    def step(self):
        """Makes one step through all the kernels and store the data."""
        order = self.update_order(len(self.kernels))
        for idx in range(self.num_kernels):
            kernel = self.kernels[order[idx]]

            for _ in range(self.inner_iterations):
               fit = kernel.fit.fitnesses
               self.archive.add(fit)

              # if lazy, we do not remove the current observed kernel
              # in this case it does not converge
               if not self.lazy:
                   if fit in self.layer:
                       self.layer.remove(fit)

               try:
                   # use the ask and tell interface to do an iteration of the
                   # kernel and store the values
                   offspring = kernel.ask()
                   offspring_values = [self.evaluate(child) for child in offspring]
                   # hypervolume_improvement corresponds to uhvi (u for
                   # uncrowded)
                   hypervolume_improvements = [self.layer.hypervolume_improvement(
                        point) for point in offspring_values]
                   self.archive.add_list(offspring_values)
                   kernel.tell(offspring, [-float(u) for u in hypervolume_improvements])

                   # allows to plot all the cma
                   if 1 > 0:
                       kernel.logger.add()

                   # compute the ratio of non dominated [offspring + mean] and
                   # store it
                   temp_archive = NDA(offspring_values, self.reference_point)
                   temp_archive.add(fit)
                   kernel.ratio_nondominated_offspring += [
                        len(temp_archive) / (1+self.num_offspring)]
               except:
                   continue

                # removing the "soon to be old" parent in the lazy case
                if fit in self.layer:
                    self.layer.remove(fit)

                # updating the fitness and add it to self.layer
                kernel.fit.fitnesses = self.evaluate(kernel.mean)
                self.layer.add(kernel.fit.fitnesses)

        # store the data
        self.hv += [self.layer.hypervolume]
        self.hv_archive += [self.archive.hypervolume]
        self.ratio_nondominated_kernels += [len(self.layer)/self.num_kernels]
        tab = [kernel.ratio_nondominated_offspring[-1]
               for kernel in self.kernels]
        percentile_tab = np.percentile(tab, [25, 50, 75])
        self.ratio_nondominated_first_quartile_offspring += [percentile_tab[0]]
        self.ratio_nondominated_median_offspring += [percentile_tab[1]]
        self.ratio_nondominated_third_quartile_offspring += [percentile_tab[2]]

    def add_kernels(self, numbers, sigma0):
        """Add 'numbers' kernels with initial mean chosen randomly around a kernel
        chosen randomly and initial sigma 'sigma0'."""
        tab = np.random.randint(0, self.num_kernels, numbers)

        for i in range(numbers):
            # compute randomly the initial mean of the new kernel
            kernel = self.kernels[tab[i]]
            lbounds = kernel.mean - 1/2 * kernel.sigma
            rbounds = kernel.mean + 1/2 * kernel.sigma
            x0 = lbounds + np.random.rand(self.dim)*(rbounds-lbounds)

            # create the kernel and add it to the kernels
            new_kernel = cma.CMAEvolutionStrategy(x0, sigma0, {'verb_filenameprefix': str(
                self.num_kernels+1), 'verbose': -9})
            new_kernel.fit.fitnesses = self.evaluate(new_kernel.mean)
            self.kernels += [new_kernel]

        # update the number of kernels
        self.num_kernels += numbers

    def run(self, budget):
        """Do as many steps as possible within the allocated budget."""
        # maxiter is the maximum number of iterations based on the budget 
        maxiter = np.int((budget-self.num_kernels)//(self.num_kernels *
                                                     self.inner_iterations*(self.num_offspring+1)))
        for l in range(maxiter):
            self.step()
            if not (l % (max(1, maxiter//10))) and budget > 2000:
                print("{}".format(l/maxiter), end=' ')

        def incremental_runs(self, budget, sigma0):
        """
        An algorithm with increasing kernels.

        We first run the algorithm until all kernels become non dominated, then
        we do two iterations and we add another kernel. We start another loop
        until the budget is consumed.
        """

        while budget - self.counteval > 0:
            maxevals = 2 * self.num_kernels * (self.num_offspring + 1)
            # the first condition corresponds to having at least one mean of a
            # kernel dominated by the others
            while self.num_kernels > len(self.layer) and budget > self.counteval:
                self.step()
            if budget > self.counteval:
                # maxevals makes the algorithm to run twice when possible, when
                # the kernels are not dominated 
                self.run(min(maxevals, budget - self.counteval))
                self.add_kernels(1, sigma0)

            print("{} kernels, {}/{} evals".format(self.num_kernels,
                                                   self.counteval, budget))

    def plot_front(self, titlelabelsize=18, axislabelsize=16):
        if self.hv != []:

            fun = self.objective_functions
            plt.figure()
            f1 = np.array([fun[0](mycma.mean) for mycma in self.kernels])
            f2 = np.array([fun[1](mycma.mean) for mycma in self.kernels])
            plt.grid(which="major")
            plt.grid(which="minor")

            plt.plot(f1, f2, 'o')
        #    plt.text(0.1,0.6,'hv_max = {}'.format(10**(-16)+float(max(self.hv))), fontsize=axislabelsize-2)
            plt.xlabel('first objective function', fontsize=axislabelsize)
            plt.ylabel('second objective function', fontsize=axislabelsize)
        #    plt.axes().set_aspect('equal')
            plt.title("front of {}, {}D, {} kernels".format(self.name,
                                                            self.dim, self.num_kernels), fontsize=titlelabelsize)

    def plot_archive(self, titlelabelsize=18, axislabelsize=16):
        if self.hv_archive != []:
            plt.figure()
            f1 = np.array([vec[0] for vec in self.archive])
            f2 = np.array([vec[1] for vec in self.archive])
            plt.grid(which="major")
            plt.grid(which="minor")

            plt.plot(f1, f2, 'o')
            plt.text(0.1, 0.6, 'hvarchive_max = {}'.format(
                float(max(self.hv_archive))), fontsize=axislabelsize-2)
            plt.xlabel('first objective function', fontsize=axislabelsize)
            plt.ylabel('second objective function', fontsize=axislabelsize)
        #    plt.axes().set_aspect('equal')
            plt.title("archive of {}, {}D, {} kernels".format(self.name,
                                                              self.dim, self.num_kernels), fontsize=titlelabelsize)

    def plot_convergence_gap(self, length=None, titlelabelsize=18, axislabelsize=16):

        plt.figure()
        maxiter = (self.counteval-self.num_kernels)//(
            self.num_kernels*self.inner_iterations*(self.num_offspring+1))
        axis = np.linspace(self.num_kernels*(
            self.kernels[0].popsize+1), maxiter*self.num_kernels*(
            self.kernels[0].popsize+1), maxiter)/self.num_kernels
        plt.grid(which="major")
        plt.grid(which="minor")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
    #    newaxis = [u for u in myaxis if u < 3000]
     #   plt.semilogy(newaxis,[float(max(self.hv))-float(u)+10**(-16) for u in self.hv[:len(newaxis)]],'-')

        if not length:
            length = max(axis) + 1
        myaxis = [u for u in axis if u < length]
        axlen = len(myaxis)
        plt.semilogy(myaxis, [float(max(self.hv))-float(u)
                              for u in self.hv[:axlen]], '-')
        plt.text(axlen/7, float(max(self.hv))-float(self.hv[0]), 'hv_max = {}'.format(
            float(max(self.hv))), fontsize=axislabelsize-2)
        plt.xlabel('function evaluations / number of kernels',
                   fontsize=axislabelsize)
        plt.ylabel('hv_max - hv', fontsize=axislabelsize)
        plt.title("COMO-CMA-ES, {}, {}D,{} kernels".format(self.name,
                                                           self.dim, self.num_kernels), fontsize=titlelabelsize-2)

    def plot_archive_gap(self, length=None,  titlelabelsize=18, axislabelsize=16):

        plt.figure()
        maxiter = (self.counteval-self.num_kernels)//(
            self.num_kernels*self.inner_iterations*(self.num_offspring+1))
        axis = np.linspace(self.num_kernels*(
            self.kernels[0].popsize+1), maxiter*self.num_kernels*(
            self.kernels[0].popsize+1), maxiter)/self.num_kernels
        plt.grid(which="major")
        plt.grid(which="minor")
        if not length:
            length = max(axis) + 1
        myaxis = [u for u in axis if u < length]
        axlen = len(myaxis)
        plt.semilogy(myaxis, [float(max(self.hv_archive))-float(u)
                              for u in self.hv_archive[:axlen]], '-')
        plt.text(axlen/7, float(max(self.hv_archive))-float(self.hv_archive[0]), 'hvarchive_max = {}'.format(
            float(max(self.hv_archive))), fontsize=axislabelsize-2)
        plt.xlabel('function evaluations / number of kernels',
                   fontsize=axislabelsize)
        plt.ylabel('hvarchive_max - hv_archive', fontsize=axislabelsize)
        plt.title("COMO-CMA-ES, {}, {}D,{} kernels".format(self.name,
                                                           self.dim, self.num_kernels), fontsize=titlelabelsize-2)

    def plot_ratios(self, length=None, titlelabelsize=18, axislabelsize=16):

        plt.figure()
        maxiter = (self.counteval-self.num_kernels)//(
            self.num_kernels*self.inner_iterations*(self.num_offspring+1))
    #        axis_offspring = np.linspace(self.num_offspring+1,maxiter*self.num_kernels*(
    #                self.num_offspring+1),maxiter*self.num_kernels)/self.num_kernels
        axis = np.linspace(self.num_kernels*(
            self.kernels[0].popsize+1), maxiter*self.num_kernels*(
            self.num_offspring+1), maxiter)/self.num_kernels
       #       plt.grid(which = "major")
        plt.grid(which="minor")
        if not length:
            length = max(axis) + 1
        myaxis = [u for u in axis if u < length]
        axlen = len(myaxis)

       #     newaxis = [u for u in myaxis if u < 2780]

        plt.plot(myaxis, self.ratio_nondominated_kernels[:axlen], 'r--',
                 label="ratio of non-dominated parents")

    #        for kernel in self.kernels:
    #        plt.plot(axis_offspring, kernel.ratio_nondominated_offspring,'g-')
        plt.plot(myaxis, self.ratio_nondominated_first_quartile_offspring[:axlen],
                 'b--', label="first quartile ratio of non-dom offspring")
        plt.plot(myaxis, self.ratio_nondominated_median_offspring[:axlen],
                 'k--', label="median ratio of non-dom offspring")
        plt.plot(myaxis, self.ratio_nondominated_third_quartile_offspring[:axlen],
                 'g--', label="third quartile ratio of non-dom offspring")
        plt.xlabel('function evaluations / num_kernels',
                   fontsize=axislabelsize)
        plt.ylabel('ratio of non-dominated points', fontsize=axislabelsize)
        plt.title("COMO-CMA-ES, {}, {}D, {} kernels".format(self.name,
                                                            self.dim, self.num_kernels), fontsize=titlelabelsize-2)
        plt.legend()

    def plot_kernels(self, numbers=3, font=plt.rcParams['font.size']):
        assert numbers < self.num_kernels + 1
        plt.figure()
        plt.rcParams['font.size'] = font

        tab = random.sample(range(self.num_kernels), numbers)

        for i in range(len(tab)):
            kernel = self.kernels[tab[i]]
            kernel.logger.plot()

    def plot_stds(self, numbers, font=plt.rcParams['font.size']):
        assert numbers < self.num_kernels + 1
        plt.figure()
        plt.rcParams['font.size'] = font
        tab = random.sample(range(self.num_kernels), numbers)
        for i in range(len(tab)):
            data = cma.CMADataLogger("{}".format(tab[i])).load()
            data.plot_stds()
#            data.plot_axes_scaling()

    def plot_axes_lengths(self, numbers, font=plt.rcParams['font.size']):
        assert numbers < self.num_kernels + 1
        plt.figure()
        plt.rcParams['font.size'] = font
        tab = random.sample(range(self.num_kernels), numbers)
        for i in range(len(tab)):
            data = cma.CMADataLogger("{}".format(tab[i])).load()
            data.plot_axes_scaling()

if __name__ == "__main__":

    dim = 10
#    num_kernels = np.int(10**3)
    num_kernels = 3

    myproblem = problem(dim, name="cigtab")
    myproblem.sep(0)
  #  myproblem.two()
    fun = myproblem.objective_functions()
    lbounds = -0*np.ones(dim)
    rbounds = 1*np.ones(dim)
    sigma0 = 0.2
#    sigma0 = np.sqrt(dim)
  #  refpoint = [1, 1]
    refpoint = [1.1, 1.1]
    budget = 3000*num_kernels

    if 1 > 0:
        mymo = CoMoCmaes(fun, dim, sigma0, lbounds, rbounds, num_kernels, refpoint, budget,
                         num_offspring=None, name=myproblem.name,
                         update_order=lambda x: np.random.permutation(x), inner_iterations=1)
    #    mymo.run(budget)
