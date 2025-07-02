
"""
This module contains the implementation of the Multiobjective framework called
Sofomore, and its instantiation with cma-es to obtain COMO-CMA-ES, defined in 
the paper [Toure, Cheikh, et al. "Uncrowded Hypervolume Improvement: 
        COMO-CMA-ES and the Sofomore framework." 
        GECCO'19-Genetic and Evolutionary Computation Conference. 2019.].

Only the bi-objective framework is functional and has been thoroughly tested.


:Author: Cheikh Toure and Nikolaus Hansen, 2019

:License: BSD 3-Clause, see LICENSE file.

"""
if 11 < 3:  # turn off infinite precision (which can be very slow in long runs)
    import moarchiving
    moarchiving.moarchiving.BiobjectiveNondominatedSortedList.hypervolume_final_float_type = float
    moarchiving.moarchiving.BiobjectiveNondominatedSortedList.hypervolume_computation_float_type = float

from . import como, sofomore_logger, como_logger, hv, nondominatedarchive

from .como import (Sofomore, IndicatorFront, get_cmas, CmaKernel, FitFun, 
                   sort_random, sort_decreasing, sort_even_odds, sort_increasing,
                   sort_odds_even,
                   RampUpSelector, GetKernelPopsizeIncrementer,
                   get_kernel_best_chv_restart, get_kernel_random_restart,
                   )

from .como import __author__, __license__, __version__

from .sofomore_logger import SofomoreDataLogger

from .nondominatedarchive import NonDominatedList

from .como_logger import COMOPlot

