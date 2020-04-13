#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cma
from cma import interfaces
import os
import matplotlib.pyplot as plt
import ast
import warnings

class SofomoreDataLogger(interfaces.BaseDataLogger):
    """data logger for class `CMAEvolutionStrategy`.

    The logger is identified by its name prefix and (over-)writes or
    reads according data files. Therefore, the logger must be
    considered as *global* variable with unpredictable side effects,
    if two loggers with the same name and on the same working folder
    are used at the same time.

    Examples
    ========
    ::

        import cma
        es = cma.CMAEvolutionStrategy(...)
        logger = cma.CMADataLogger().register(es)
        while not es.stop():
            ...
            logger.add()  # add can also take an argument

        logger.plot() # or a short cut can be used:
        cma.plot()  # plot data from logger with default name

        logger2 = cma.CMADataLogger('just_another_filename_prefix').load()
        logger2.plot()
        logger2.disp()

        import cma
        from matplotlib.pylab import *
        res = cma.fmin(cma.ff.sphere, rand(10), 1e-0)
        logger = res[-1]  # the CMADataLogger
        logger.load()  # by "default" data are on disk
        semilogy(logger.f[:,0], logger.f[:,5])  # plot f versus iteration, see file header
        cma.s.figshow()

    Details
    =======
    After loading data, the logger has the attributes `xmean`, `xrecent`,
    `std`, `f`, `D` and `corrspec` corresponding to ``xmean``,
    ``xrecentbest``, ``stddev``, ``fit``, ``axlen`` and ``axlencorr``
    filename trails.

    :See: `disp` (), `plot` ()
    """
    default_prefix = 'outsofomore' + os.sep
    # default_prefix = 'outcmaes'
    # names = ('axlen','fit','stddev','xmean','xrecentbest')
    # key_names_with_annotation = ('std', 'xmean', 'xrecent')

    def __init__(self, name_prefix=default_prefix, modulo=1, append=False):
        """initialize logging of data from a `CMAEvolutionStrategy`
        instance, default ``modulo=1`` means logging with each call

        """
        # super(CMAData, self).__init__({'iter':[], 'stds':[], 'D':[],
        #        'sig':[], 'fit':[], 'xm':[]})
        # class properties:
#        if isinstance(name_prefix, CMAEvolutionStrategy):
#            name_prefix = name_prefix.opts.eval('verb_filenameprefix')
        if name_prefix is None:
            name_prefix = SofomoreDataLogger.default_prefix
        self.name_prefix = os.path.abspath(os.path.join(*os.path.split(name_prefix)))
        if name_prefix is not None and name_prefix.endswith((os.sep, '/')):
            self.name_prefix = self.name_prefix + os.sep
        self.file_names = ('hypervolume', 'hypervolume_archive', 'len_archive', 'ratio_active_kernel',
                           'ratio_nondom_incumb', 'ratio_nondom_offsp_incumb',
                           'median_sigmas', 'median_axis_ratios', 'median_min_stds',
                           'median_max_stds', 'median_stds')
        self.modulo = modulo
        """how often to record data, allows calling `add` without args"""
        self.append = append
        """append to previous data"""
        self.counter = 0
        """number of calls to `add`"""
        self.last_iteration = 0
        self.registered = False
        self.persistent_communication_dict = cma.utilities.utils.DictFromTagsInString()

    def register(self, es, append=None, modulo=None):
        """register a `Sofomore` instance for logging,
        ``append=True`` appends to previous data logged under the same name,
        by default previous data are overwritten.

        """
#        if not isinstance(es, CMAEvolutionStrategy):
#            utils.print_warning("""only class CMAEvolutionStrategy should
#    be registered for logging. The used "%s" class may not to work
#    properly. This warning may also occur after using `reload`. Then,
#    restarting Python should solve the issue.""" %
#                                str(type(es)))
        self.es = es
        if append is not None:
            self.append = append
        if modulo is not None:
            self.modulo = modulo
        self.registered = True
        return self

    def initialize(self, modulo=None):
        """reset logger, overwrite original files, `modulo`: log only every modulo call"""
        if modulo is not None:
            self.modulo = modulo
        try:
            es = self.es  # must have been registered
        except AttributeError:
            pass  # TODO: revise usage of es... that this can pass
            raise AttributeError('call register() before initialize()')

        self.counter = 0  # number of calls of add
        self.last_iteration = 0  # some lines are only written if iteration>last_iteration
        if self.modulo <= 0:
            return self

        # create path if necessary
        if os.path.dirname(self.name_prefix):
            try:
                os.makedirs(os.path.dirname(self.name_prefix))
            except OSError:
                pass  # folder exists

        # write headers for output
                
        fn = self.name_prefix + 'median_sigmas.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, median sigmas, ' +
                        ', ' + 
                        '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)
            
        fn = self.name_prefix + 'median_axis_ratios.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, median axis ratios, ' +
                        ', ' + 
                        '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)
            
        fn = self.name_prefix + 'median_min_stds.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, median min stds, ' +
                        ', ' + 
                        '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)

        fn = self.name_prefix + 'median_max_stds.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, median max stds, ' +
                        ', ' + 
                        '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)
            
        fn = self.name_prefix + 'median_stds.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, median stds, ' +
                '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)

        fn = self.name_prefix + 'hypervolume.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, hypervolume, ' +
                        ', ' + 
                        '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)

        fn = self.name_prefix + 'hypervolume_archive.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, hypervolume of archive' +
                        '\n')
        except (IOError, OSError):
            print('could not open/write file ' + fn)


        fn = self.name_prefix + 'len_archive.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iter, evals, length archive" ' +
                        ', ' + 
                        '\n')
        except (IOError, OSError):
            print('could not open/write file ' + fn)
            
        fn = self.name_prefix + 'ratio_inactive_kernels.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, ratio active kernels, ' +
                        '\n')
        except (IOError, OSError):
            print('could not open/write file ' + fn)

        fn = self.name_prefix + 'ratio_nondom_incumb.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, ratio nondominated incumbents, ' +
                        '\n')
        except (IOError, OSError):
            print('could not open/write file ' + fn)


        fn = self.name_prefix + 'ratio_nondom_offsp_incumb.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iter, evals, first quartile nondom ' + 
                        'offspring and incumbent, median nondom offspring and ' +
                        'incumbent, last quartile nondom offspring and incumbent' +
                        ', ' +
                        '\n')
        except (IOError, OSError):
            print('could not open/write file ' + fn)

        return self
    # end def __init__
              
    def add(self, es=None, more_data=(), modulo=None):
        """append some logging data from `Sofomore` class instance `es`,
        if ``number_of_times_called % modulo`` equals to zero, never if ``modulo==0``.

        ``more_data`` is a list of additional data to be recorded where each
        data entry must have the same length.

        When used for a different optimizer class, this function can be
        (easily?) adapted by changing the assignments under INTERFACE
        in the implemention.

        """
        mod = modulo if modulo is not None else self.modulo
        self.counter += 1
        if mod == 0 or (self.counter > 3 and (self.counter - 1) % mod):
            return
        if es is None:
            try:
                es = self.es  # must have been registered
            except AttributeError :
                raise AttributeError('call `add` with argument `es` or ``register(es)`` before ``add()``')
        elif not self.registered:
            self.register(es)

        if self.counter == 1 and not self.append and self.modulo != 0:
            self.initialize()  # write file headers
            self.counter = 1

        # --- INTERFACE, can be changed if necessary ---
#        if not isinstance(es, CMAEvolutionStrategy):  # not necessary
#            utils.print_warning('type CMAEvolutionStrategy expected, found '
#                                + str(type(es)), 'add', 'CMADataLogger')
        evals = es.countevals
        iteration = es.countiter
        hypervolume = float(es.pareto_front_cut.hypervolume)
        hypervolume_archive = 0.0
        len_archive = 0
        if es.isarchive:
            hypervolume_archive = float(es.archive.hypervolume)
            len_archive = len(es.archive)
        ratio_inactive = 1 - len(es._active_indices) / len(es)
        ratio_nondom_incumbent = len(es.pareto_front_cut)/len(es)
                
        for i in range(len(es.offspring)):
            idx = es.offspring[i][0]
            kernel = es.kernels[idx]
            
            temp_archive = es.NDA(kernel._last_offspring_f_values, es.reference_point)
            temp_archive.add(kernel.objective_values)
            es._ratio_nondom_offspring_incumbent[idx] = len(temp_archive) / (
                    1 + len(kernel._last_offspring_f_values) )
                
    
        first_quartile_ratio_offspring_incumbent = np.percentile(es._ratio_nondom_offspring_incumbent, 25);
        median_ratio_offspring_incumbent = np.percentile(es._ratio_nondom_offspring_incumbent, 50);
        last_quartile_ratio_offspring_incumbent = np.percentile(es._ratio_nondom_offspring_incumbent, 75);
        
        
        median_axis_ratios = np.median([kernel.D.max() / kernel.D.min() \
        if not kernel.opts['CMA_diagonal'] or kernel.countiter > kernel.opts['CMA_diagonal']
        else max(kernel.sigma_vec*1) / min(kernel.sigma_vec*1) for kernel in es.kernels])
        median_sigmas = np.median([kernel.sigma for kernel in es.kernels])
        median_min_stds = np.median([kernel.sigma * min(kernel.sigma_vec * kernel.dC**0.5) \
                                     for kernel in es.kernels])
        median_max_stds = np.median([kernel.sigma * max(kernel.sigma_vec * kernel.dC**0.5) \
                                     for kernel in es.kernels])
        median_stds = self.es.median_stds

        
        # --- end interface ---

        try:
            # median axis ratios
            fn = self.name_prefix + 'median_axis_ratios.dat'
            with open(fn, 'a') as f:
                f.write(str(iteration) + ' '
                        + str(evals) + ' '
                        + str(median_axis_ratios)
                        + '\n')
            
            # median sigmas
            fn = self.name_prefix + 'median_sigmas.dat'
            with open(fn, 'a') as f:
                f.write(str(iteration) + ' '
                        + str(evals) + ' '
                        + str(median_sigmas)
                        + '\n')
            
            # median min stds
            fn = self.name_prefix + 'median_min_stds.dat'
            with open(fn, 'a') as f:
                f.write(str(iteration) + ' '
                        + str(evals) + ' '
                        + str(median_min_stds)
                        + '\n')
                
            # median max stds
            fn = self.name_prefix + 'median_max_stds.dat'
            with open(fn, 'a') as f:
                f.write(str(iteration) + ' '
                        + str(evals) + ' '
                        + str(median_max_stds)
                        + '\n')

            # median stds
            fn = self.name_prefix + 'median_stds.dat'
            with open(fn, 'a') as f:
                f.write(str(iteration) + ' '
                        + str(evals) + ' '
                        + str(median_stds)
                        + '\n')            
            # hypervolume
            if iteration > self.last_iteration:
                fn = self.name_prefix + 'hypervolume.dat'
                with open(fn, 'a') as f:
                    f.write(str(iteration) + ' '
                            + str(evals) + ' '
                            + str(hypervolume) + ' '
                            + '\n')
 
           # hypervolume archive
            if iteration > self.last_iteration:
                fn = self.name_prefix + 'hypervolume_archive.dat'
                with open(fn, 'a') as f:
                    f.write(str(iteration) + ' '
                            + str(evals) + ' '
                            + str(hypervolume_archive)
                            + '\n')

            # length archive
            if iteration > self.last_iteration:
                fn = self.name_prefix + 'len_archive.dat'
                with open(fn, 'a') as f:
                    f.write(str(iteration) + ' '
                            + str(evals) + ' '
                            + str(len_archive)
                            + '\n')
        # ratio of inactive kernels
            if iteration > self.last_iteration:
                fn = self.name_prefix + 'ratio_inactive_kernels.dat'
                with open(fn, 'a') as f:
                    f.write(str(iteration) + ' '
                            + str(evals) + ' '
                            + str(ratio_inactive)
                            + '\n')
                # ratio of non dominated incumbents
            if iteration > self.last_iteration:
                fn = self.name_prefix + 'ratio_nondom_incumb.dat'
                with open(fn, 'a') as f:
                    f.write(str(iteration) + ' '
                            + str(evals) + ' '
                            + str(ratio_nondom_incumbent)
                            + '\n')
            # ratio of nondominated [incumbent + its offspring]
            if iteration > self.last_iteration:
                fn = self.name_prefix + 'ratio_nondom_offsp_incumb.dat' 
                with open(fn, 'a') as f:
                    f.write(str(iteration) + ' '
                            + str(evals) + ' '
                            + str(first_quartile_ratio_offspring_incumbent) + ' '
                            + str(median_ratio_offspring_incumbent) + ' '
                            + str(last_quartile_ratio_offspring_incumbent)
                            + '\n')
            

        except (IOError, OSError):
            pass

        self.last_iteration = iteration

        
    def load(self, filenames):
        """
        """
        iteration = []
        countevals = []
        if isinstance(filenames, str):
            filenames = [filenames]
        res = []
        for i in range(len(filenames)):
            filename = filenames[i]
            with open(filename) as f:
                tab = [line.rstrip() for line in f.readlines()[1:]] #the first 
                # line of our file is a headline
                maxsplit = 2 if filename[-15:] == 'median_stds.dat' else -1
                newtab = [list(map(ast.literal_eval,line.split(maxsplit = maxsplit))) for line in tab]
                length = len(newtab[0])
                for k in range(2, length):
                    res += [np.array([line[k] for line in newtab])]
                    
                if i == 0: # we define iteration, countevals just for the first filename
                    iteration = np.array([line[0] for line in newtab])
                    countevals = np.array([line[1] for line in newtab])
                
        return iteration, countevals, res
        
    def plot(self, filename, x_iteration = 0):
        """
        """
        iteration, countevals, res = self.load(filename)
        for i in range(len(res)):   
            if not x_iteration:
                plt.plot(countevals, res)
            else:
                plt.plot(iteration, res)
            
    def plot_front(self, aspect=None):
        """
        """
        if aspect is not None:
            myaxes = plt.gca()
            try:
                myaxes.set_aspect(aspect) # usually, aspect = 'equal'
            except:
                pass
        moes = self.es
        try:
            plt.plot([u[0] for u in moes.archive], [u[1] for u in moes.archive], '.',
                     label = "archive")
        except:
            pass
        plt.plot([u[0] for u in moes.pareto_front_cut], [u[1] for u in moes.pareto_front_cut], 'o',
                 label = "cma-es incumbents")
        pass
     #   plt.legend()
    def plot_ratios(self, iabscissa=1):
        
        """
        """
        
        # also put tolx/median(max_stds)
        
        from matplotlib import pyplot
        fn_incumbent = self.name_prefix + 'ratio_nondom_incumb.dat'
        fn_inactive = self.name_prefix + 'ratio_inactive_kernels.dat'
        fn_nondom = self.name_prefix + 'ratio_nondom_offsp_incumb.dat' 
        filenames = fn_incumbent, fn_inactive, fn_nondom
        iteration, countevals, res = self.load(filenames)
        absciss = countevals if iabscissa else iteration
        self._enter_plotting()
  #      color = iter(pyplot.cm.plasma_r(np.linspace(0.35, 1, 3)))
        self._xlabel(iabscissa)
        mylabel = ['ratio nondom incumbents', 'ratio inactive kernels',
                   '1st quartile ratio nondom off+incumb',
                   'median ratio nondom off+incumb',
                   '3rd quartile ratio nondom off+incumb']
        for i in range(5):
            pyplot.plot(absciss, res[i], label = mylabel[i])
          #  pyplot.plot(absciss, res[i],
           #             '-', color=next(color), label = mylabel[i])
    #        pyplot.semilogy(absciss, res[i],
     #                       '-', color=next(color), label = mylabel[i])
        # pyplot.hold(True)
        pyplot.grid(True)
        ax = np.array(pyplot.axis())
        # ax[1] = max(minxend, ax[1])
        pyplot.axis(ax)
        # pyplot.title('')
        pyplot.legend()
        # pyplot.xticks(xticklocs)
        self._xlabel(iabscissa)
        self._finalize_plotting()
        return self
    
    def plot_divers(self, iabscissa=1):
        """
        """
        # also put tolx/median(max_stds)
 
        from matplotlib import pyplot
        fn_axis_ratios = self.name_prefix + 'median_axis_ratios.dat'
        fn_max_stds = self.name_prefix + 'median_max_stds.dat'
        fn_min_stds = self.name_prefix + 'median_min_stds.dat' 
        fn_sigmas = self.name_prefix + 'median_sigmas.dat'
        fn_hypervolume = self.name_prefix + 'hypervolume.dat'
        fn_archive = self.name_prefix + 'hypervolume_archive.dat' 
        fn_len_archive = self.name_prefix + 'len_archive.dat' 

        # we call `self.load` twice: for the median-related files and for the 
        # hypervolume related ones: because the iteration and countevals might 
        # be different, depending on the value of `iteration > self.last_iteration`
        # inside the `add` method.
        filenames_median = (fn_axis_ratios, fn_max_stds, fn_min_stds, fn_sigmas)
        (iteration_median, countevals_median, 
         res_median) = self.load(filenames_median)
        absciss_median = countevals_median if iabscissa else iteration_median
        
        filenames_hypervolume = (fn_hypervolume, fn_archive, fn_len_archive)
        (iteration_hypervolume, countevals_hypervolume, 
         res_hypervolume) = self.load(filenames_hypervolume)
        absciss_hypervolume = (countevals_hypervolume if iabscissa 
                               else iteration_hypervolume)

        self._enter_plotting()
  #      color = iter(pyplot.cm.plasma_r(np.linspace(0.35, 1, 3)))
        self._xlabel(iabscissa)
        mylabel = ['median axis ratios', 'median max stds',
                   'median min stds', 'median sigmas',
                   'convergence gap', 'archive gap', 'inverse length archive']
        for i in range(4):
            pyplot.semilogy(absciss_median, res_median[i], label = mylabel[i])
          #  pyplot.plot(absciss, res[i],
           #             '-', color=next(color), label = mylabel[i])
    #        pyplot.semilogy(absciss, res[i],
     #                       '-', color=next(color), label = mylabel[i])
        # pyplot.hold(True)
        offset_convergence_gap = self.es.best_hypervolume_pareto_front
        pyplot.semilogy(absciss_hypervolume, [offset_convergence_gap - u 
                                          for u in res_hypervolume[0]],
                    label = mylabel[4], nonposy = 'clip')
        current_archive = res_hypervolume[1]
        try:
            offset_archive_gap = current_archive[-1]
            pyplot.semilogy(absciss_hypervolume, 
                            [offset_archive_gap - u for u in current_archive], 
                            label = mylabel[5], nonposy = 'clip')
        except IndexError:
            warnings.warn("empty archive")
        pyplot.semilogy(absciss_hypervolume, [1/u for u in res_hypervolume[2]], 
                    label = mylabel[6])
        pyplot.grid(True)
        ax = np.array(pyplot.axis())
        # ax[1] = max(minxend, ax[1])
        pyplot.axis(ax)
        # pyplot.title('')
        pyplot.legend()
        # pyplot.xticks(xticklocs)
        self._xlabel(iabscissa)
        self._finalize_plotting()
        return self
    
        
    def plot_stds(self, iabscissa=1):
        
        """
        """
        
        from matplotlib import pyplot
        filename = self.name_prefix + 'median_stds.dat'
        iteration, countevals, res = self.load(filename)
        absciss = countevals if iabscissa else iteration
        self._enter_plotting()
  #      color = iter(pyplot.cm.plasma_r(np.linspace(0.35, 1, 3)))
        self._xlabel(iabscissa)
        
        pyplot.semilogy(absciss, res[0])
          #  pyplot.plot(absciss, res[i],
           #             '-', color=next(color), label = mylabel[i])
    #        pyplot.semilogy(absciss, res[i],
     #                       '-', color=next(color), label = mylabel[i])
        # pyplot.hold(True)
        pyplot.grid(True)
        ax = np.array(pyplot.axis())
        # ax[1] = max(minxend, ax[1])
        pyplot.axis(ax)
        # pyplot.title('')
    #    pyplot.legend()
        # pyplot.xticks(xticklocs)
        self._xlabel(iabscissa)
        pyplot.title("median (sorted) standard deviations in all coordinates")
        self._finalize_plotting()
        return self
        
    def _enter_plotting(self, fontsize=7):
        """assumes that a figure is open """
        from matplotlib import pyplot
        # interactive_status = matplotlib.is_interactive()
        self.original_fontsize = pyplot.rcParams['font.size']
        # if font size deviates from default, we assume this is on purpose and hence leave it alone
        if pyplot.rcParams['font.size'] == pyplot.rcParamsDefault['font.size']:
            pyplot.rcParams['font.size'] = fontsize
        # was: pyplot.hold(False)
        # pyplot.gcf().clear()  # opens a figure window, if non exists
        pyplot.ioff()
    def _finalize_plotting(self):
        from matplotlib import pyplot
        pyplot.subplots_adjust(left=0.05, top=0.96, bottom=0.07, right=0.95)
        # pyplot.tight_layout(rect=(0, 0, 0.96, 1))
        pyplot.draw()  # update "screen"
        pyplot.ion()  # prevents that the execution stops after plotting
        pyplot.show()
        pyplot.rcParams['font.size'] = self.original_fontsize
    def _xlabel(self, iabscissa=1):
        from matplotlib import pyplot
        pyplot.xlabel('iterations' if iabscissa == 0
                      else 'function evaluations')

            
            
            
            