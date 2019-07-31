#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Cheikh and ...
"""


#import warnings
#import fractions
import numpy as np
import cma
from moarchiving import BiobjectiveNondominatedSortedList as NDA
#import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
#from math import inf
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
                 inner_iterations=1,
                 lazy=False,
                 name=None,
                 add_method=lambda *args: None,
                 test_method=lambda *args: None
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
        - 'positive integer' inner_iterations : number of iterations of each
        CMAES in one step
        - 'bool' lazy : if true, the kernel objective value is not removed
        during its update
        - 'string' name : the name of the objective function
        - 'fun' add_method : the function which is called when adding kernels
        - 'fun' test_method : a function which is called at each step
        """
        self.objective_functions = objective_functions
        self.dim = dim
        self.num_kernels = num_kernels
        self.sigma0 = sigma0
        self.reference_point = reference_point
        self.counteval = 0
        self.countstep = 0
        self.lazy = lazy
        # initialization of kernels
        if kernels is None:
            kernels = []
            for i in range(num_kernels):
                x0 = lbounds + np.random.rand(self.dim)*(rbounds-lbounds)
                kernels += [cma.CMAEvolutionStrategy(x0, sigma0, {'verb_filenameprefix': str(
                    i), 'conditioncov_alleviate': [np.inf, np.inf],
                    'CMA_const_trace': 'True', 'verbose':-1})]
        self.kernels = kernels
        
        # definition of num_offspring: number of offspring for each kernel
        self.num_offspring = self.kernels[0].popsize
        
        self.max_evaluations = max_evaluations
        self.inner_iterations = inner_iterations
        self.update_order = update_order
        self.name = name
        self.add_method = add_method
        self.test_method = test_method

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
        self.num_kernels_step = []  # store the number of kernels at the end of
                                    # each step
        self.counteval_step = []   # store the number of eval at the end of
                                   # each step

    def evaluate(self, x_var):
        """Evaluate the objective functions on the decision variable x_var and 
        increment the number of evaluations.
        """
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
        self.counteval_step += [self.counteval]
        self.num_kernels_step += [self.num_kernels]
        self.countstep += 1

        # call the test_method
        self.test_method(self)

    def add_kernel(self, x0, sigma0):
        "Add a kernel of mean x0 and initial stepsize sigma0 to self."""
        prop = {'verb_filenameprefix': str(self.num_kernels+1),
                'conditioncov_alleviate': [np.inf, np.inf],
                'CMA_const_trace': 'True',
                'verbose': -1}
        new_kernel = cma.CMAEvolutionStrategy(x0, sigma0, prop)
        new_kernel.fit.fitnesses = self.evaluate(new_kernel.mean)
        new_kernel.ratio_nondominated_offspring = []
        self.kernels += [new_kernel]
        self.num_kernels += 1

    def run(self, budget):
        """Do as many steps as possible within the allocated budget."""
        bound = budget / 10
        level = 0
        while self.counteval < budget:
            self.step()
            if self.counteval > (level + 1) * bound:
                level = int(self.counteval / budget * 10)
                print("We are at ", level, "/10")

    def incremental_runs(self, budget):
        """
        An algorithm with increasing kernels.

        We first run the algorithm until all kernels become non dominated, then
        we do two iterations and we add another kernel. We start another loop
        until the budget is consumed.
        """

        while budget - self.counteval > 0:
            # the first condition corresponds to having at least one mean of a
            # kernel dominated by the others or not dominating the reference point
            while self.num_kernels > len(self.layer) and budget > self.counteval:
                self.step()
            if budget > self.counteval:
                # maxevals makes the algorithm run twice when possible, when
                # the kernels are not dominated 
                self.step()
                self.step()
                self.add_method(self)
            print("{} kernels, {}/{} evals, {} steps".format(self.num_kernels,
                                                   self.counteval, budget,
                                                   len(self.counteval_step)))

    def _save(self, plot_name=None, end=None):
        """Save the figure with the name
        '{plot_name}_{self.name}_{self.counteval}_{end}.png'
        in the directory '{self.name}_{self.counteval}_{end}'"""
        # create the name of the file and the directory
        fig_name = plot_name
        dir_name = ""
        if self.name is not None:
            fig_name += "_" + self.name
            dir_name += self.name
        fig_name += "_" + str(self.counteval)
        dir_name += "_" + str(self.counteval)
        if end is not None:
            fig_name += "_" + end
            dir_name += "_" + end
        fig_name += ".png"
        dir_name += "/"

        here = os.getcwd()
        results_dir = os.path.join(here, dir_name)

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        plt.savefig(dir_name + fig_name)

    def plot_front(self, titlelabelsize=18, axislabelsize=16, save=False,
                   end=None):
        """
        Plot the objective values of the incumbents of the kernels.
        """
        fun = self.objective_functions
        plt.figure()
        # defining the x-values (f1) and the y-values (f2)
        f1 = np.array([fun[0](mycma.mean) for mycma in self.kernels])
        f2 = np.array([fun[1](mycma.mean) for mycma in self.kernels])
        
        plt.grid(which="major")
        plt.grid(which="minor")

        plt.plot(f1, f2, 'o')
        plt.xlabel('first objective function', fontsize=axislabelsize)
        plt.ylabel('second objective function', fontsize=axislabelsize)
        plt.title("front of {}, {}D, {} kernels".format(self.name,
                                                        self.dim, self.num_kernels), fontsize=titlelabelsize)

        # save the plot
        if save:
            self._save("front", end)

    def plot_archive(self, titlelabelsize=18, axislabelsize=16, save=False,
                     end=None):
        """
        Plot the objective values of all non-dominated points evaluated so far.
        """
        if self.hv_archive != []: # at least one step is done to the CoMoCmaes instance
            plt.figure()
            f1 = np.array([vec[0] for vec in self.archive])
            f2 = np.array([vec[1] for vec in self.archive])
            plt.grid(which="major")
            plt.grid(which="minor")

            plt.plot(f1, f2, 'o')

            # we print 'max(self.hv_archive)' where 'self.hv_archive' is nonempty:
            plt.text(0.1, 0.6, 'hvarchive_max = {}'.format(
                float(max(self.hv_archive))), fontsize=axislabelsize-2)

            plt.xlabel('first objective function', fontsize=axislabelsize)
            plt.ylabel('second objective function', fontsize=axislabelsize)
            plt.title("archive of {}, {}D, {} kernels".format(self.name,
                                                              self.dim, self.num_kernels), fontsize=titlelabelsize)
        else:
            print("No step is done on {}.".format(self.name))

        # save the plot
        if save:
            self._save("archive", end)

    def _plot_gap(self, what, length, titlelabelsize, axislabelsize, save, end):
        """
        Plot the gap between 'what' (which is either hv or hv_archive and has
        only negative values) and its maximum (in log scale) for the first
        'length' values of 'what'.
        """
        plt.figure()
        plt.grid(which="major")
        plt.grid(which="minor")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        if what == "hv":
            value_max = float(max(self.hv))
            values = [value_max-float(u) for u in self.hv[:length]]
            length_max = len(self.hv)
            if length is None:
                length = length_max
            length = min(length, length_max)
        if what == "hv_archive":
            value_max = float(max(self.hv_archive))
            values = [value_max-float(u) for u in
                      self.hv_archive[:length]]
            length_max = len(self.hv_archive)
            if length is None:
                length = length_max
            length = min(length, length_max)

        axis = self.counteval_step[:length]
        plt.semilogy(axis, values, '-')
        step_incr = [i for i in range(1, self.countstep) if self.num_kernels_step[i] >
                     self.num_kernels_step[i - 1]]
        for step in step_incr:
            plt.scatter(self.counteval_step[step-1], values[step-1],
                        color='r', marker='x')

        # print the value of the offset hv_max = max(self.hv) somewhere likely to be visible:
        plt.text(length/7, values[0], (what + '_max = {}').format(
            value_max), fontsize=axislabelsize-2)

        plt.xlabel('function evaluations', fontsize=axislabelsize)
        plt.ylabel(what + "_max - " + what, fontsize=axislabelsize)
        plt.title("COMO-CMA-ES, {}, {}D,{} kernels".format(self.name, self.dim,
                    self.num_kernels), fontsize=titlelabelsize-2)

        # save the plot
        if save:
            self._save(what + "gap", end)

    def plot_convergence_gap(self, length=None, titlelabelsize=18,
                             axislabelsize=16, save=False, end=None):
        """
        Plot the convergence gap (in log scale): 'max(self.hv))-self.hv[k]' for
        k = 0,...,length-1.
        """
        self._plot_gap("hv", length, titlelabelsize, axislabelsize, save, end)

    def plot_archive_gap(self, length=None,  titlelabelsize=18,
                         axislabelsize=16, save=False, end=None):
        """
        Plot the archive gap (in log scale): 'max(self.hv_archive))-self.hv_archive[k]' for
        k = 0,...,length-1.
        """
        self._plot_gap("hv_archive", length, titlelabelsize, axislabelsize, save, end)

    def plot_ratios(self, length=None, titlelabelsize=18, axislabelsize=16,
                    save=False, end=None):
        """
        Plot the statistics of the ratios of non-dominated offspring + incumbent,
        and the ratio of non-dominated incumbents.
        """
        length_max = len(self.ratio_nondominated_kernels)
        if length is None:
            length = length_max
        else:
            length = min(length, length_max)

        plt.figure()
        myaxis = self.counteval_step[:length]
        plt.grid(which="minor")

        plt.plot(myaxis, self.ratio_nondominated_kernels[:length], 'r--',
                 label="ratio of non-dominated parents")
        plt.plot(myaxis, self.ratio_nondominated_first_quartile_offspring[:length],
                 'b--', label="first quartile ratio of non-dom offspring")
        plt.plot(myaxis, self.ratio_nondominated_median_offspring[:length],
                 'k--', label="median ratio of non-dom offspring")
        plt.plot(myaxis, self.ratio_nondominated_third_quartile_offspring[:length],
                 'g--', label="third quartile ratio of non-dom offspring")
        plt.xlabel('function evaluations',
                   fontsize=axislabelsize)
        plt.ylabel('ratio of non-dominated points', fontsize=axislabelsize)
        plt.title("COMO-CMA-ES, {}, {}D, {} kernels".format(self.name,
                                                            self.dim,
                                                            self.num_kernels),
                  fontsize=titlelabelsize-2)
        plt.legend()

        # save the plot
        if save:
            self._save("ratios", end)

    def plot_kernels(self, numbers=1, font=plt.rcParams['font.size'],
                     save=False, end=None):
        """
        Choose uniformly at random 'numbers' kernels and plot them using the CMA logger tools.
        - font is the font size of the plots.
        """
        assert numbers < self.num_kernels + 1
        plt.figure()
        plt.rcParams['font.size'] = font 

        # sample randomly at uniform 'numbers' points in {0,...,self.num_kernels-1}:
        tab = random.sample(range(self.num_kernels_step[-1]), numbers)

        for i in range(len(tab)):
            kernel = self.kernels[tab[i]]
            kernel.logger.plot() # plot 'kernel'

        # save the plot
        if save:
            self._save("kernels", end)

    def plot_stds(self, numbers=1, font=plt.rcParams['font.size'], save=False,
                  end=None):
        """
        Choose uniformly at random 'numbers' kernels and plot their standards deviations
        divided by the step-size in all coordinates, using the CMADataLogger tools.
        - font is the font size of the plots.
        """
        assert numbers < self.num_kernels + 1
        plt.figure()
        plt.rcParams['font.size'] = font
        
        # sample randomly at uniform 'numbers' points in {0,...,self.num_kernels-1}:
        tab = random.sample(range(self.num_kernels_step[-1]), numbers)
        
        for i in range(len(tab)):
            data = cma.CMADataLogger("{}".format(tab[i])).load() # load the data to be plotted
            data.plot_stds() # plot the data

        # save the plot
        if save:
            self._save("stds", end)


    def plot_axes_lengths(self, numbers=1, font=plt.rcParams['font.size'],
                          save=False, end=None):
        """
        Choose uniformly at random 'numbers' kernels and plot their covariance matrices 
        square root eigenvalues using the CMADataLogger tools.
        - font is the font size of the plots.
        """
        assert numbers < self.num_kernels + 1
        plt.figure()
        plt.rcParams['font.size'] = font

        # sample randomly at uniform 'numbers' points in {0,...,self.num_kernels-1}:
        tab = random.sample(range(self.num_kernels_step[-1]), numbers)

        for i in range(len(tab)):
            data = cma.CMADataLogger("{}".format(tab[i])).load() # load the data to be plotted
            data.plot_axes_scaling() # plot the data

        # save the plot
        if save:
            self._save("axes_lengths", end)

    def plot_all(self, save=False, end=None):
        print(1)
        self.plot_front(save=save, end=end)
        print(2)
        self.plot_archive(save=save, end=end)
        print(3)
        self.plot_convergence_gap(save=save, end=end)
        print(4)
        self.plot_archive_gap(save=save, end=end)
        print(5)
        self.plot_ratios(save=save, end=end)
        print(6)
        self.plot_kernels(save=save, end=end)
        print(7)
        self.plot_stds(save=save, end=end)
        print(8)
        self.plot_axes_lengths(save=save, end=end)


def add_kernel_close(self):
    """Add 'numbers' kernels with initial mean chosen randomly around a kernel
    chosen randomly and initial sigma 'sigma0'.
    """
    idx = np.random.randint(0, self.num_kernels)

    # compute randomly the initial mean of the new kernel
    kernel = self.kernels[idx]
    lbounds = kernel.mean - 1/2 * kernel.sigma
    rbounds = kernel.mean + 1/2 * kernel.sigma
    x0 = lbounds + np.random.rand(self.dim)*(rbounds-lbounds)
    self.add_kernel(x0, kernel.sigma)

def add_kernels_middle(self, part):
    """
    Add around a fraction 'part' of kernels with mean in the middle of
    already existing kernel ND means, ordered by fitnesses.
    Others infos are copied from the left kernel.
    """
    kernels_sorted = sorted(self.kernels, key=lambda kernel: kernel.fit.fitnesses)
    nb = max(int(part * (self.num_kernels - 1)), 1)
    tab = np.random.randint(0, self.num_kernels-1, nb)

    for i in range(nb):
        idx = tab[i]
        ker1 = kernels_sorted[idx]
        ker2 = kernels_sorted[idx + 1]
        if ker1.fit.fitnesses in self.layer and ker2.fit.fitnesses in self.layer:
            prop = {'verb_filenameprefix': str(self.num_kernels+1),
                    'conditioncov_alleviate': [np.inf, np.inf],
                    'CMA_const_trace': 'True',
                    'verbose': -1}
            ker = ker1._copy_light(prop)
            ker.mean = (ker1.mean + ker2.mean) / 2
            ker.fit.fitnesses = self.evaluate(ker.mean)
            ker.ratio_nondominated_offspring = []
            self.kernels += [ker]
            self.num_kernels += 1

def check_kernels_middle_nd(self):
    """Check the ratio of points in the middle of ND points which are non-dominated."""
    kernels_sorted = sorted(self.kernels, key=lambda kernel: kernel.fit.fitnesses)
    ratio = 0
    nb = 0
    for idx in range(self.num_kernels - 1):
        ker1 = kernels_sorted[idx]
        ker2 = kernels_sorted[idx + 1]
        nd_fit = self.layer
        if ker1.fit.fitnesses in nd_fit and ker2.fit.fitnesses in nd_fit:
            nb += 1
            x0 = (ker1.mean + ker2.mean) / 2
            if not self.layer.dominates(self.evaluate(x0)):
                ratio += 1
            self.counteval -= 1 # compensation
    if nb != 0 and ratio != nb:
        ratio /= nb
        print("the ratio of nd points in the middle of nd neighbours is < 1:", ratio)

if __name__ == "__main__":
    dim = 10

    # def of the function
    def sphere(x, x0):
        # for simplicity, x0 is a scalar
        return(sum([(elt - x0)**2 for elt in x]))

    mypb = problem(dim, name="sphere")

    b_simple = 0
    if not b_simple:
        fun = mypb.objective_functions()
        name = mypb.name
    if b_simple:
        fun = (lambda x: sphere(x, 0), lambda x: sphere(x, 1))
        name = "double-sphere"

    inner_iterations = 1
    sigma0 = 0.1
    lbounds, rbounds = -1, 3
    num_kernels = 10
    refpoint = [10, 10]
    budget = 20000
    add_method = lambda y: add_kernels_middle(y, 0.1)
    #add_method = lambda y: add_kernels_middle_copy(y, 0.1)
    test_method = check_kernels_middle_nd
    for i in range(1):
        mymo = CoMoCmaes(fun, dim, sigma0, lbounds, rbounds, num_kernels, refpoint,
                     budget, name=name, update_order=lambda x: np.random.permutation(x),
                     inner_iterations=inner_iterations, add_method=add_method,
                         test_method=test_method)
        #mymo.run(budget)
        mymo.incremental_runs(budget)
        mymo.plot_all(save=True, end="incr")
    mpl.pyplot.show(block=True)

#    dim = 10
#    num_kernels = 11
#
#    # problem is a class of bi-objective convex quadratic problems
#    # from the module 'problems'.
#    myproblem = problem(dim, name="cigtab")
#    myproblem.sep(0) # we are in the 'sep-0' case
#  #  myproblem.two()
#    fun = myproblem.objective_functions()
#    lbounds = -0*np.ones(dim)
#    rbounds = 1*np.ones(dim)
#    sigma0 = 0.2
#    refpoint = [1.1, 1.1]
#    budget = 3000*num_kernels
#
#    if 1 > 0:
#        mymo = CoMoCmaes(fun, dim, sigma0, lbounds, rbounds, num_kernels, refpoint, budget,
#                         name=myproblem.name,
#                         update_order=lambda x: np.random.permutation(x), inner_iterations=1)
#    #    mymo.run(budget)
