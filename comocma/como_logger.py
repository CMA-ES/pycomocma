import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from time import time
import ast
import warnings

class COMOPlot:
    '''
    TODO :
    - Maybe we should write a warning message if we detect that the store function is modified after some storing has already been done ? 
    - Correct bugs in plot_hvi function -- check if the error message still appear
    - Add the equivalent of the 4th Niko's plot
    - Add dominated kernels in plot_archive. 
    - Add sigmas (first, maximum and last) on a plot. 
    - Add condition number on iterations plot.
    - Add convergence speed plot ?
    '''
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

    def store0(self, name, v, overwrite=False, init=None):
        '''
        Store the value v in the file name.txt.
        If 'erase' is True, the file is emptied before writing in it. 
        It adds the 'init' value at the beginning of the file, when the file is empty (incompatible with overwrite='True'). 
        Remark : still works when v is a list. 
        '''
        if overwrite:
            with open(self.dir + name + '.txt', 'w') as f:
                f.write("%s\n" % v)
        else:
            with open(self.dir + name + '.txt', 'a') as f:
                if os.stat(self.dir + name + '.txt').st_size == 0 and init is not None: # if the file is empty
                    f.write("%s\n" % init)
                f.write("%s\n" % v)
    
    def store(self, moes):
        "Store data on moes using store0."
        # list of non-dominated final incumbents
        ND_finalincumbents = moes.NDA([kernel.objective_values for kernel in moes.kernels[:-1]], moes.reference_point)
        ND_allincumbents = moes.NDA([kernel.objective_values for kernel in moes.kernels if kernel.objective_values is not None], moes.reference_point)
        # data stored at the beginning of new run - except the first one
        if moes.kernels[-1].countevals == 0 and len(moes.kernels) > 1:
            # store the kind of restart used for the new kernel
            self.store0("kindstart", moes.kernels[-1]._rampup_method.__name__, init="initial start")
            # store data relative to stopping criterion at the end of the last run
            # store the stopping criterion of the last completed run
            self.store0("stopcrit", moes.kernels[-2].stop())
            # store the tolx value at end of the last completed run
            self.store0("tolx", moes.kernels[-2].stop(get_value='tolx'))
            # store the tolfunrel criterion constant of the last completed run
            self.store0("tolfunrel_const", moes.kernels[-2].fit.median0 - moes.kernels[-2].fit.median_min)
            # store the number of runs which have been completed
            self.store0("last_completedrun", len(moes.kernels) - 1, overwrite=True)
            # store the number of dominated final incumbents
            self.store0("num_dominatedfinalincumbents", len(moes.kernels) - 1 - len(ND_finalincumbents))
            # store the condition number
            self.store0("conditionnumber", moes.kernels[-2].sm.condition_number)

            # store the number of iterations done so far
            self.store0("iter_newrun", moes.countiter, init=0)
            # store the time index
            self.store0("time", time())
        # data stored at the end of each iteration - TODO: to be completed
        # store the hypervolume of the archive
        self.store0("hv_archive", float(moes.archive.hypervolume))
        self.store0("hv_incumbents", float(ND_allincumbents.hypervolume))
        # store the objective values of the kernel incumbents
        self.store0("objective_values", [k.objective_values for k in moes])
        # store only the last archive
        self.store0("last_archive", moes.archive, overwrite=True)
        # update the number of times this function have been called
        self.num_calls += 1

        # call the storing functions
        for fun in self.storing_funs:
            fun(self, moes)
    
    def load(self):
        '''
        Load the data stored in the directory self.dir if it has not been done yet and store it in self.data as a dictionary.
        Return this dictionary.

        If a file name begins by "last_" and the file contains only one value, it is assumed that the intent is not to store all the history but only the last value. 
        '''
        if self.num_data == self.num_calls: # the stored data has already been read entirely and stored in self.data
            return self.data
        dic = dict()
        for file in os.listdir(self.dir):
            with open(self.dir + file) as f:
                lines = f.read().split("\n")[:-1] # the list of lines content, without the \n symbols and the last empty line
                key = file[:-4] # remove the .txt
                try:
                    dic[key] = [ast.literal_eval(l) for l in lines]
                except:
                    dic[key] = lines
                if file[:5] == "last_" and len(lines) == 1: # if we store only the last value
                    dic[key] = dic[key][0] # dic[key] is not a list containing this one value but the value itself
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
        try:
            n_runs = dic["last_completedrun"]
        except:
            warnings.warn("Since no CMA-ES run has been completed yet, the proportion of dominated final incumbents was not plotted.")
            return
        plt.figure()
        plt.plot(range(1,n_runs+1), [dic["num_dominatedfinalincumbents"][i] / (i+1) for i in range(n_runs)], '.')
        plt.xlabel("number of runs completed")
        plt.ylabel("proportion of dominated final incumbents")
        plt.title("proportion of final incumbents which are now dominated")
        plt.grid()

    def plot_iterations_per_restart(self):
        '''Plot the number of iterations per restart.'''
        dic = self.load()
        try:
            n_runs = dic["last_completedrun"]
        except:
            warnings.warn("Since no CMA-ES run has been completed yet, the number of iterations per restart was not plotted.")
            return
        plt.figure()
        legend = []
        for kindstart in set(dic["kindstart"]):
            plt.plot([i+1 for i in range(n_runs) if dic["kindstart"][i]==kindstart],[dic["iter_newrun"][i+1] - dic["iter_newrun"][i]  for i in range(n_runs) if dic["kindstart"][i]==kindstart], '.')
            legend.append(kindstart)
        plt.legend(legend)
        plt.xlabel("number of runs completed")
        plt.ylabel("number of iterations of the last run")
        plt.title("Number of iterations per run")
        plt.grid()

    def plot_convergence_speed(self):
        '''Plot the convergence speed.'''
        dic = self.load()
        if self.offset is None:
            offset = dic["hv_archive"][-1]
        else:
            offset = self.offset
        n_iters = len(dic["hv_incumbents"])
        plt.figure()
        plt.semilogy(range(1, n_iters + 1),[offset - dic["hv_incumbents"][i] for i in range(n_iters)], 'g')
        plt.semilogy(range(1, n_iters + 1),[offset - dic["hv_archive"][i] for i in range(n_iters)], 'b')
        plt.legend(["$S=$final incumbents only", "$S=$archive"])
        plt.xlabel("iterations")
        plt.ylabel("offset - $HV_r(S)$ ")
        plt.title("Convergence plot (offset=%.9e)" %offset)
        plt.grid(which="both")

    def plot_archive(self):
        '''Plot the archive.'''
        dic = self.load()
        non_dominated_kernels = [v for v in dic["objective_values"][-1] if v in dic["last_archive"]]
        plt.suptitle('Archive represented in the objective space')
        plt.title('HV archive=%.9e' % (dic["hv_archive"][-1]), fontsize=10)
        plt.xlabel("first objective function")
        plt.ylabel("second objective function")
        xy = np.asarray(dic["last_archive"])
        len(xy) and plt.plot(xy[:,0], xy[:,1], '.')
        xy = np.asarray(non_dominated_kernels)
        len(xy) and plt.plot(xy[:,0], xy[:,1], '.')
        plt.legend(["archive (" + str(len(dic["last_archive"])) + ")", 
                    "final incumbents of CMA-ES runs \n not dominated by the archive (" + str(len(non_dominated_kernels)) + ")"])
        plt.grid()
    
    def plot_hvi(self):
        '''Plot information regarding hypervolume improvement.'''
        dic = self.load()
        try:
            n_runs = dic["last_completedrun"]
        except:
            warnings.warn("Since no CMA-ES run has been completed yet, the number of iterations per restart was not plotted.")
            return

        # plot the hvi lines for archive and incumbents
        plt.figure()
        plt.semilogy(range(1,n_runs+1), [dic["hv_archive"][dic["iter_newrun"][i+1]] - dic["hv_archive"][dic["iter_newrun"][i]] for i in range(n_runs)], 'lightblue', linestyle='--')
        plt.semilogy(range(1,n_runs+1), [dic["hv_incumbents"][dic["iter_newrun"][i+1]] - dic["hv_incumbents"][dic["iter_newrun"][i]] for i in range(n_runs)], 'lightgreen', linestyle='--')

        # plot the dots which correspond to each kind of restart in a different color
        legend = ['hvi archive', 'hvi final incumbents']
        for kindstart in set(dic["kindstart"]):
            if kindstart != "initial start":
                idx = [i for i in range(n_runs) if dic["kindstart"][i]==kindstart]
                hvi_archive = [dic["hv_archive"][dic["iter_newrun"][i+1]] - dic["hv_archive"][dic["iter_newrun"][i]] for i in idx]
                hvi_incumbents = [dic["hv_incumbents"][dic["iter_newrun"][i+1]] - dic["hv_incumbents"][dic["iter_newrun"][i]] for i in idx]
                idxx = [i+1 for i in idx] # when counting the starts, we start at 1
                plt.semilogy(idxx + idxx, hvi_archive + hvi_incumbents, '.')
                legend.append(kindstart)
        plt.xlabel("runs")
        plt.ylabel("hvi")
        plt.grid(which="both")
        plt.legend(legend)
        
        


