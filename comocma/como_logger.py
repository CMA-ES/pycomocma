import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from time import time
import ast

class COMOPlot_Callback:
    def __init__(self, storing_funs=[]):
        '''
        It takes a facultative parameter:
        - 'storing_funs': a list of functions which will be called at the end of any self.store call
        '''
        # create the directory where the data will be stored
        path = os.getcwd()
        name_save = datetime.datetime.now().strftime("%y%m%d_%Hh%Mm%Ss")
        path1 = path + "/OutputCOMO/"
        path2 = path1 + name_save + "/"
        if not os.path.exists(path1):
            os.mkdir(path1)
        if not os.path.exists(path2):
            os.mkdir(path2)
        else:
            i = 1
            while not os.path.exists(path1 + name_save + str(i) + "/"):
                i += 1
            path2 = path1 + name_save + str(i) + "/"
            os.mkdir(path2)
        # remember in which directory the data is stored
        self.dir = path2
        # initialization of the offset for convergence speed plotting, which is ideally the hypervolume of the Pareto front
        self.offset = None
        # remember a list of storing functions which will be called with arguments self and moes on top of the standard storing function
        # each time the function self.store is called
        self.storing_funs = storing_funs
        # dictionary which will later contain the data
        self.data = None
        # number of times the function store have been called
        self.num_calls = 0
        # last call stored in self.data
        # when self.num_data < self.num_calls, it means that the data in self.data is not up to date
        self.num_data = 0

    def store0(self, name, v):
        """Store the value v in the file name.txt ."""
        with open(self.dir + name + '.txt', 'a') as f:
            f.write("%s\n" % v)
    
    def store(self, moes):
        "Store data on moes using store0."
        # list of non-dominated final incumbents
        ND = moes.NDA([kernel.objective_values for kernel in moes.kernels[:-1]], moes.reference_point)
        # data stored at the beginning of new run - except the first one
        if moes.kernels[-1].countevals == 0 and len(moes.kernels) > 1:
            # store the kind of restart used for the new kernel
            if moes.kernels[-1].objective_values is None:
                kind_restart = "random"
            else:
                kind_restart = "best_chv"
            self.store0("kindrestart", kind_restart)
            # store data relative to stopping criterion at the end of the last run
            # store the stopping criterion of the last completed run
            self.store0("stopcrit", moes.kernels[-2].stop())
            # store the tolx value at end of the last completed run
            self.store0("tolx", moes.kernels[-2].stop(get_value='tolx'))
            # store the tolfunrel criterion constant of the last completed run
            self.store0("tolfunrel_const", moes.kernels[-2].fit.median0 - moes.kernels[-2].fit.median_min)
            # store the number of runs which have been completed
            self.store0("num_completedruns", len(moes.kernels) - 1)
            # store the number of dominated final incumbents
            self.store0("num_dominatedfinalincumbents", len(moes.kernels) - 1 - len(ND))
            # store the condition number
            self.store0("conditionnumber", moes.kernels[-2].sm.condition_number)

            # store the number of iterations done so far
            self.store0("iter_newrun", moes.countiter)
            # store the time index
            self.store0("time", time())
        # data stored at the end of each iteration - TODO: to be completed
        # store the hypervolume of the archive
        self.store0("hv_archive", float(moes.archive.hypervolume))
        self.store0("hv_incumbents", float(ND.hypervolume))
        # update the number of times this function have been called
        self.num_calls += 1

        # call the storing functions
        for fun in self.storing_funs:
            fun(self, moes)
    
    def load(self):
        '''
        Load the data stored in the directory self.dir if it has not been done yet and store it in self.data as a dictionary.
        Return this dictionary. 
        '''
        if self.num_data == self.num_calls: # the stored data has already been read entirely and stored in self.data
            return self.data
        dic = dict()
        for file in os.listdir(self.dir):
            with open(self.dir + file) as f:
                lines = f.readlines()
                key = file[:-4] # remove the .txt
                try:
                    dic[key] = [ast.literal_eval(l) for l in lines]
                except:
                    dic[key] = lines
        self.data = dic
        self.num_data = self.num_calls
        return dic
    
    def plot_everything(self):
        """Call every method of the class whose name begins by 'plot_' and which is not 'plot_everything' itself."""
        fun_names = [name for name in dir(self) if name[:5]=='plot_' and name!= 'plot_everything']
        for name in fun_names:
            getattr(self, name)()

    def plot_proportion_dominated_final_incumbents(self):
        '''Plot the proportion of dominated final incumbents.'''
        dic = self.load()
        n_runs = dic["num_completedruns"][-1]
        plt.figure()
        plt.plot(dic["num_completedruns"], [dic["num_dominatedfinalincumbents"][i] / (i+1) for i in range(n_runs)], '.')
        plt.xlabel("number of runs completed")
        plt.ylabel("proportion of dominated final incumbents")
        plt.title("proportion of final incumbents which are now dominated")

    def plot_iterations_per_restart(self):
        '''Plot the number of iterations per restart.'''
        dic = self.load()
        n_runs = dic["num_completedruns"][-1]
        plt.figure()
        plt.plot([dic["num_completedruns"][i+1] for i in range(n_runs-1) if dic["kindrestart"][i]=="best_chv\n"],[dic["iter_newrun"][i+1] - dic["iter_newrun"][i]  for i in range(n_runs - 1) if dic["kindrestart"][i]=="best_chv\n"], '.')
        plt.plot([dic["num_completedruns"][i+1] for i in range(n_runs-1) if dic["kindrestart"][i]=="random\n"],[dic["iter_newrun"][i+1] - dic["iter_newrun"][i]  for i in range(n_runs - 1) if dic["kindrestart"][i]=="random\n"], '.')
        plt.legend(["best_chv restart", "random restart"])
        plt.xlabel("number of runs completed")
        plt.ylabel("number of iterations of the last run")
        plt.title("Number of iterations per run")

    def plot_convergence_speed(self):
        '''Plot the convergence speed.'''
        dic = self.load()
        if self.offset is None:
            self.offset = dic["hv_archive"][-1]
        n_iters = len(dic["hv_incumbents"])
        plt.figure()
        plt.semilogy(range(1, n_iters + 1),[self.offset - dic["hv_incumbents"][i] for i in range(n_iters)], 'g')
        plt.semilogy(range(1, n_iters + 1),[self.offset - dic["hv_archive"][i] for i in range(n_iters)], 'b')
        plt.semilogy(dic["iter_newrun"], [self.offset - dic["hv_incumbents"][i] for i in range(n_iters) if i in dic["iter_newrun"]], '.g')
        plt.semilogy(dic["iter_newrun"], [self.offset - dic["hv_archive"][i] for i in range(n_iters) if i in dic["iter_newrun"]], '.b')
        plt.legend(["$S=$final incumbents only", "$S=$archive"])
        plt.xlabel("iterations")
        plt.ylabel("offset - $HV_r(S)$ ")
        plt.title("Convergence plot")
