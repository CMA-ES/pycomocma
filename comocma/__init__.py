
from . import como, sofomore_logger, hv, nondominatedarchive

from .como import (Sofomore, IndicatorFront, get_cmas, CmaKernel, FitFun, 
                   sort_random, sort_decreasing, sort_even_odds, sort_increasing,
                   sort_odds_even)
from .sofomore_logger import SofomoreDataLogger

from .nondominatedarchive import NonDominatedList

__author__ = 'Cheikh Toure and Nikolaus Hansen'
__version__ = "0.5.0"
