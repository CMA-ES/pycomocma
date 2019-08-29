#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cma
from cma import interfaces
import os
import time

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
                           'ratio_nondom_incumb', 'nondom_offsp_incumb',
                           'median_sigmas', 'median_axis_ratios', 'median_min_stds',
                           'median_max_stds')
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
                
        strseedtime = 'seed=%s, %s' % (str(es.opts['seed']), time.asctime())

        fn = self.name_prefix + 'median_sigmas.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, median sigmas, ' +
                        strseedtime +
                        ', ' + 
                        '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)
            
        fn = self.name_prefix + 'median_axis_ratios.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, median axis ratios, ' +
                        strseedtime +
                        ', ' + 
                        '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)
            
        fn = self.name_prefix + 'median_min_stds.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, median min stds, ' +
                        strseedtime +
                        ', ' + 
                        '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)

        fn = self.name_prefix + 'median_max_stds.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, median max stds, ' +
                        strseedtime +
                        ', ' + 
                        '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)

        fn = self.name_prefix + 'hypervolume.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, hypervolume, ' +
                        strseedtime +
                        ', ' + 
                        '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)

        fn = self.name_prefix + 'hypervolume_archive.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, hypervolume of archive' +
                        strseedtime +
                        '\n')
        except (IOError, OSError):
            print('could not open/write file ' + fn)


        fn = self.name_prefix + 'len_archive.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iter, evals, length archive" ' +
                        strseedtime +
                        ', ' + 
                        '\n')
        except (IOError, OSError):
            print('could not open/write file ' + fn)
            
        fn = self.name_prefix + 'ratio_inactive_kernels.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, ratio active kernels, ' +
                        strseedtime +
                        '\n')
        except (IOError, OSError):
            print('could not open/write file ' + fn)

        fn = self.name_prefix + 'ratio_nondom_incumb.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, ratio nondominated incumbents, ' +
                        strseedtime +
                        '\n')
        except (IOError, OSError):
            print('could not open/write file ' + fn)


        fn = self.name_prefix + 'nondom_offsp_incumb.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iter, evals, nondom offspring and incumbents" ' +
                        strseedtime +
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
        hypervolume = float(es.front.hypervolume)
        hypervolume_archive = 0.0
        len_archive = 0
        if es.active_archive:
            hypervolume_archive = float(es.archive.hypervolume)
            len_archive = len(es.archive)
        ratio_inactive = es.ratio_inactive
        ratio_nondom_incumbent = len(es.front)/es.num_kernels
        
        temp_archive = es.nda(es._last_fvalues, es.reference_point)
        ratio_offspring_incumbent = len(temp_archive)/len(es._last_fvalues)

        
        median_axis_ratios = np.median([kernel.D.max() / kernel.D.min() \
        if not kernel.opts['CMA_diagonal'] or kernel.countiter > kernel.opts['CMA_diagonal']
        else max(kernel.sigma_vec*1) / min(kernel.sigma_vec*1) for kernel in es.kernels])
        median_sigmas = np.median([kernel.sigma for kernel in es.kernels])
        median_min_stds = np.median([kernel.sigma * min(kernel.sigma_vec * kernel.dC**0.5) \
                                     for kernel in es.kernels])
        median_max_stds = np.median([kernel.sigma * max(kernel.sigma_vec * kernel.dC**0.5) \
                                     for kernel in es.kernels])

        
        # --- end interface ---

        try:
            # median axis ratios
            fn = self.name_prefix + 'median_axis_ratios.dat'
            with open(fn, 'a') as f:
                f.write(str(iteration) + ' '
                        + str(evals) + ' '
                        + str(median_axis_ratios) + ' '
                        + '\n')
            
            # median sigmas
            fn = self.name_prefix + 'median_sigmas.dat'
            with open(fn, 'a') as f:
                f.write(str(iteration) + ' '
                        + str(evals) + ' '
                        + str(median_sigmas) + ' '
                        + '\n')
            
            # median min stds
            fn = self.name_prefix + 'median_min_stds.dat'
            with open(fn, 'a') as f:
                f.write(str(iteration) + ' '
                        + str(evals) + ' '
                        + str(median_min_stds) + ' '
                        + '\n')
                
            # median max stds
            fn = self.name_prefix + 'median_max_stds.dat'
            with open(fn, 'a') as f:
                f.write(str(iteration) + ' '
                        + str(evals) + ' '
                        + str(median_max_stds) + ' '
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
                            + str(ratio_nondom_incumbent) + ' '
                            + '\n')
            # ratio of nondominated [offspring + incumbent]
            if iteration > self.last_iteration:
                fn = self.name_prefix + 'nondom_offsp_incumb.dat' 
                with open(fn, 'a') as f:
                    f.write(str(iteration) + ' '
                            + str(evals) + ' '
                            + str(ratio_offspring_incumbent) + ' '
                            + '\n')
            

                
        except (IOError, OSError):
            pass

        self.last_iteration = iteration

        
        
        