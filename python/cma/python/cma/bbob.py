#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cocoex

class NormalizedCocoProblem:
    """A translated and shifted coco problem with all-zeros as
  
    global minimum and 0 as optimal function value.
  
    Example:
      
    >>> import cocoex
    >>> bbob = cocoex.Suite('bbob', '', '')
    >>> dimension = 10
    >>> f6 = NormalizedCocoProblem(bbob, 6, dimension)
    >>> assert f6(dimension * [0]) == 0

    """
    def __init__(self, suite, fun_nb, dimension, instance=1):
       """return a "normalized" coco problem.
      
       `suite` must be a `cocoex.Suite` instance, the further
       parameters `fun_nb`, `dimension`, `instance` are passed to
       `suite.get_problem_by_function_dimension_instance`.

       Details
       -------
       The original coco problem resides in attribute `f`.

       The returned "normalized" version of the function is useful
       to make single experiments with algorithms which are invariant
       under x- and f-translations. The output of the experiments
       is (much) easier to interpret for the normalized version.
       """
       if not hasattr(suite, 'get_problem_by_function_dimension_instance'):
           suite = cocoex.Suite(suite, '', '')
       self.f = suite.get_problem_by_function_dimension_instance(
                    fun_nb, dimension, instance)
       self.f._best_parameter('print')
       self._xopt = np.loadtxt('._bbob_problem_best_parameter.txt')
       self._fopt = self.f(self._xopt)
       self._inherit_docstrings(self.f)
       self._inherit_attributes(self.f)
       
    def __call__(self, x):
       """Main functionality, return f(x), where ``f(zeros)==0`` is the minimizer"""
       return self.f(self._xopt + x) - self._fopt
    def constraint(self, x):
        return self.f.constraint(self._xopt + x)
    def _inherit_attributes(self, f):
        """link attributes from `f`.

        This might not always lead to the expected result.
        """
        for name in dir(f):
            if not name.startswith('_') and not hasattr(self, name):
                setattr(self, name, getattr(f, name))
    def _inherit_docstrings(self, f):
        """set method ``__doc__`` attributes from `f`"""
        for name in dir(f):
            if not name.startswith('_') and hasattr(self, name):
                if getattr(getattr(self, name), '__doc__', False) is None:
                    try:
                        setattr(getattr(NormalizedCocoProblem, name),
                                '__doc__', getattr(f, name).__doc__)
                    except AttributeError:
                        print('Docstring of % could not be inherited.' % name)

