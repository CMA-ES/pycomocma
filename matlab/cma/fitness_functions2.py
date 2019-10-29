# -*- coding: utf-8 -*-
"""versatile container for test objective functions.

For the time being this is probably best used like::

    from cma.fitness_functions2 import ff

Tested with::

    import cma, cma.fitness_functions2
    cma.ff2 = cma.fitness_functions2
    cma.ff2.__dict__
    dff, dff2 = dir(cma.ff), dir(cma.ff2)
    for f in dff2:
        if f in dff and f not in ('BBOB', 'epslow',
                                  'fun_as_arg') and not f.startswith('_'):
            try:
                getattr(cma.ff, f)(aa([1.,2,3])) == getattr(cma.ff2, f)(aa([1.,2,3]))
            except:
                try:
                    assert getattr(cma.ff, f)(aa([1.,2,3]), cma.ff.sphere)\
                               == getattr(cma.ff2, f)(aa([1.,2,3]), cma.ff2.sphere)
                except:
                    assert all(getattr(cma.ff, f)(aa([1.,2,3]), cma.ff.sphere)
                               == getattr(cma.ff2, f)(aa([1.,2,3]), cma.ff2.sphere))

"""

from __future__ import (absolute_import, division, print_function,
                        )  # unicode_literals, with_statement)
# from __future__ import collections.MutableMapping
# does not exist in future, otherwise Python 2.5 would work, since 0.91.01
__author__ = "Nikolaus Hansen"

from .utilities.python3for2 import *
del absolute_import, division, print_function

import numpy as np
# arange, cos, size, eye, inf, dot, floor, outer, zeros, linalg.eigh,
# sort, argsort, random, ones,...
from numpy import array, dot, isscalar, sum  # sum is not needed
# from numpy import inf, exp, log, isfinite
# to access the built-in sum fct:  ``__builtins__.sum`` or ``del sum``
# removes the imported sum and recovers the shadowed build-in
try: np.median([1,2,3,2])  # fails currently in pypy, also sigma_vec.scaling
except AttributeError:
    def _median(x):
        x = sorted(x)
        if len(x) % 2:
            return x[len(x) // 2]
        return (x[len(x) // 2 - 1] + x[len(x) // 2]) / 2
    np.median = _median
from .utilities.utils import rglen as _rglen

# $Source$  # according to PEP 8 style guides, but what is it good for?
# $Id: fitness_functions.py 4150 2015-03-20 13:53:36Z hansen $
# bash $: svn propset svn:keywords 'Date Revision Id' fitness_functions.py

from . import bbobbenchmarks as BBOB
from .fitness_transformations import rotate  #, ComposedFunction, Function

def _iqr(x):
    x = sorted(x)
    i1 = int(len(x) / 4)
    i3 = int(3*len(x) / 4)
    return x[i3] - x[i1]

def somenan(x, fun, p=0.1):
    """returns sometimes np.NaN, otherwise fun(x)"""
    if np.random.rand(1) < p:
        return np.NaN
    else:
        return fun(x)

def epslow(fun, eps=1e-7, Neff=lambda x: int(len(x)**0.5)):
    return lambda x: fun(x[:Neff(x)]) + eps * np.mean(x**2)

def rand(x):
    """Random test objective function"""
    return np.random.random(1)[0]
def linear(x):
    return -x[0]
def lineard(x):
    if 1 < 3 and any(array(x) < 0):
        return np.nan
    if 1 < 3 and sum([(10 + i) * x[i] for i in _rglen(x)]) > 50e3:
        return np.nan
    return -sum(x)
def sphere(x):
    """Sphere (squared norm) test objective function"""
    # return np.random.rand(1)[0]**0 * sum(x**2) + 1 * np.random.rand(1)[0]
    return sum((x + 0)**2)
def subspace_sphere(x, visible_ratio=1/2):
    """
    """
    # here we could use an init function, that is this would
    # preferably be a class
    m = int(visible_ratio * len(x) + 1)
    x = np.asarray(x)[np.random.permutation(len(x))[:m]]
    return sum(x**2)
def pnorm(x, p=0.5):
    return sum(np.abs(x)**p)**(1./p)
def grad_sphere(x, *args):
    return 2*array(x, copy=False)
def grad_to_one(x, *args):
    return array(x, copy=False) - 1
def sphere_pos(x):
    """Sphere (squared norm) test objective function"""
    # return np.random.rand(1)[0]**0 * sum(x**2) + 1 * np.random.rand(1)[0]
    c = 0.0
    if x[0] < c:
        return np.nan
    return -c**2 + sum((x + 0)**2)
def spherewithoneconstraint(x):
    return sum((x + 0)**2) if x[0] > 1 else np.nan
def elliwithoneconstraint(x, idx=[-1]):
    return ellirot(x) if all(array(x)[idx] > 1) else np.nan

def spherewithnconstraints(x):
    return sum((x + 0)**2) if all(array(x) > 1) else np.nan
# zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
def noisysphere(x, noise=2.10e-9, cond=1.0, noise_offset=0.10):
    """noise=10 does not work with default popsize, ``cma.NoiseHandler(dimension, 1e7)`` helps"""
    return elli(x, cond=cond) * np.exp(0 + noise * np.random.randn() / len(x)) + noise_offset * np.random.rand()
def spherew(x):
    """Sphere (squared norm) with sum x_i = 1 test objective function"""
    # return np.random.rand(1)[0]**0 * sum(x**2) + 1 * np.random.rand(1)[0]
    # s = sum(abs(x))
    # return sum((x/s+0)**2) - 1/len(x)
    # return sum((x/s)**2) - 1/len(x)
    return -0.01 * x[0] + abs(x[0])**-2 * sum(x[1:]**2)
def epslowsphere(x, eps=1e-7, Neff=lambda x: int(len(x)**0.5)):
    """TODO: define as wrapper"""
    return np.mean(x[:Neff(x)]**2) + eps * np.mean(x**2)
def partsphere(x):
    """Sphere (squared norm) test objective function"""
    partsphere.evaluations += 1
    # return np.random.rand(1)[0]**0 * sum(x**2) + 1 * np.random.rand(1)[0]
    dim = len(x)
    x = array([x[i % dim] for i in range(2 * dim)])
    N = 8
    i = partsphere.evaluations % dim
    # f = sum(x[i:i + N]**2)
    f = sum(x[np.random.randint(dim, size=N)]**2)
    return f
partsphere.evaluations = 0
def sectorsphere(x):
    """asymmetric Sphere (squared norm) test objective function"""
    return sum(x**2) + (1e6 - 1) * sum(x[x < 0]**2)
def cornersphere(x):
    """Sphere (squared norm) test objective function constraint to the corner"""
    nconstr = len(x) - 0
    if any(x[:nconstr] < 1):
        return np.NaN
    return sum(x**2) - nconstr
def cornerelli(x):
    """ """
    if any(x < 1):
        return np.NaN
    return elli(x) - elli(np.ones(len(x)))
def cornerellirot(x):
    """ """
    if any(x < 1):
        return np.NaN
    return ellirot(x)
def normalSkew(f):
    N = np.random.randn(1)[0]**2
    if N < 1:
        N = f * N  # diminish blow up lower part
    return N
def noiseC(x, func=sphere, fac=10, expon=0.8):
    f = func(x)
    N = np.random.randn(1)[0] / np.random.randn(1)[0]
    return max(1e-19, f + (float(fac) / len(x)) * f**expon * N)
def noise(x, func=sphere, fac=10, expon=1):
    f = func(x)
    # R = np.random.randn(1)[0]
    R = np.log10(f) + expon * abs(10 - np.log10(f)) * np.random.rand(1)[0]
    # sig = float(fac)/float(len(x))
    # R = log(f) + 0.5*log(f) * random.randn(1)[0]
    # return max(1e-19, f + sig * (f**np.log10(f)) * np.exp(R))
    # return max(1e-19, f * np.exp(sig * N / f**expon))
    # return max(1e-19, f * normalSkew(f**expon)**sig)
    return f + 10**R  # == f + f**(1+0.5*RN)
def cigar(x, rot=0, cond=1e6, noise=0):
    """Cigar test objective function"""
    if rot:
        x = rotate(x)
    x = [x] if isscalar(x[0]) else x  # scalar into list
    f = [(x[0]**2 + cond * sum(x[1:]**2)) * np.exp(noise * np.random.randn(1)[0] / len(x)) for x in x]
    return f if len(f) > 1 else f[0]  # 1-element-list into scalar
def grad_cigar(x, *args):
    grad = 2 * 1e6 * np.array(x)
    grad[0] /= 1e6
    return grad
def diagonal_cigar(x, cond=1e6):
    axis = np.ones(len(x)) / len(x)**0.5
    proj = dot(axis, x) * axis
    s = sum(proj**2)
    s += cond * sum((x - proj)**2)
    return s
def tablet(x, cond=1e6, rot=0):
    """Tablet test objective function"""
    x = np.asarray(x)
    if rot and rot is not tablet:
        x = rotate(x)
    x = [x] if isscalar(x[0]) else x  # scalar into list
    f = [cond * x[0]**2 + sum(x[1:]**2) for x in x]
    return f if len(f) > 1 else f[0]  # 1-element-list into scalar
def grad_tablet(x, *args):
    grad = 2 * np.array(x)
    grad[0] *= 1e6
    return grad
def cigtab(y):
    """Cigtab test objective function"""
    X = [y] if isscalar(y[0]) else y
    f = [1e-4 * x[0]**2 + 1e4 * x[1]**2 + sum(x[2:]**2) for x in X]
    return f if len(f) > 1 else f[0]
def cigtab2(x, condition=1e8, n_axes=None, n_short_axes=None):
    """cigtab with 5% long and short axes.

    `n_axes: int`, if not `None`, sets the number of long axes to
    `n_axes` and also the number of short axes if `n_short_axes` is
    `None`.
    """
    m = n_axes
    if m is None:
        m = max((1, int(0.05 * (len(x) + 1./2))))
    ms = n_short_axes
    if ms is None:
        ms = m
    x = np.asarray(x)
    f = sum(x[m:-ms]**2)
    f += condition**-0.5 * sum(x[:m]**2)
    f += condition**0.5 * sum(x[-ms:]**2)
    return f
def twoaxes(y):
    """Cigtab test objective function"""
    X = [y] if isscalar(y[0]) else y
    N2 = len(X[0]) // 2
    f = [1e6 * sum(x[0:N2]**2) + sum(x[N2:]**2) for x in X]
    return f if len(f) > 1 else f[0]
def ellirot(x):
    return elli(array(x), 1)
def hyperelli(x):
    N = len(x)
    return sum((np.arange(1, N + 1) * x)**2)
def halfelli(x):
    l = len(x) // 2
    felli = elli(x[:l])
    return felli + 1e-8 * sum(x[l:]**2)
def elli(x, rot=0, xoffset=0, cond=1e6, actuator_noise=0.0, both=False):
    """Ellipsoid test objective function"""
    x = np.asarray(x)
    if not isscalar(x[0]):  # parallel evaluation
        return [elli(xi, rot) for xi in x]  # could save 20% overall
    if rot:
        x = rotate(x)
    N = len(x)
    if actuator_noise:
        x = x + actuator_noise * np.random.randn(N)

    ftrue = sum(cond**(np.arange(N) / (N - 1.)) * (x + xoffset)**2) \
            if N > 1 else (x + xoffset)**2

    alpha = 0.49 + 1. / N
    beta = 1
    felli = np.random.rand(1)[0]**beta * ftrue * \
            max(1, (10.**9 / (ftrue + 1e-99))**(alpha * np.random.rand(1)[0]))
    # felli = ftrue + 1*np.random.randn(1)[0] / (1e-30 +
    #                                           np.abs(np.random.randn(1)[0]))**0
    if both:
        return (felli, ftrue)
    else:
        # return felli  # possibly noisy value
        return ftrue  # + np.random.randn()
def grad_elli(x, *args):
    cond = 1e6
    N = len(x)
    return 2 * cond**(np.arange(N) / (N - 1.)) * array(x, copy=False)
def fun_as_arg(x, *args):
    """``fun_as_arg(x, fun, *more_args)`` calls ``fun(x, *more_args)``.

    Use case::

        fmin(cma.fun_as_arg, args=(fun,), gradf=grad_numerical)

    calls fun_as_args(x, args) and grad_numerical(x, fun, args=args)

    """
    fun = args[0]
    more_args = args[1:] if len(args) > 1 else ()
    return fun(x, *more_args)

def grad_numerical(x, func, epsilon=None):
    """symmetric gradient"""
    eps = 1e-8 * (1 + abs(x)) if epsilon is None else epsilon
    grad = np.zeros(len(x))
    ei = np.zeros(len(x))  # float is 1.6 times faster than int
    for i in _rglen(x):
        ei[i] = eps[i]
        grad[i] = (func(x + ei) - func(x - ei)) / (2*eps[i])
        ei[i] = 0
    return grad
def elliconstraint(x, cfac=1e8, tough=True, cond=1e6):
    """ellipsoid test objective function with "constraints" """
    N = len(x)
    f = sum(cond**(np.arange(N)[-1::-1] / (N - 1)) * x**2)
    cvals = (x[0] + 1,
             x[0] + 1 + 100 * x[1],
             x[0] + 1 - 100 * x[1])
    if tough:
        f += cfac * sum(max(0, c) for c in cvals)
    else:
        f += cfac * sum(max(0, c + 1e-3)**2 for c in cvals)
    return f
def rosen(x, alpha=1e2):
    """Rosenbrock test objective function"""
    x = [x] if isscalar(x[0]) else x  # scalar into list
    x = np.asarray(x)
    f = [sum(alpha * (x[:-1]**2 - x[1:])**2 + (1. - x[:-1])**2) for x in x]
    return f if len(f) > 1 else f[0]  # 1-element-list into scalar
def grad_rosen(x, *args):
    N = len(x)
    grad = np.zeros(N)
    grad[0] = 2 * (x[0] - 1) + 200 * (x[1] - x[0]**2) * -2 * x[0]
    i = np.arange(1, N - 1)
    grad[i] = 2 * (x[i] - 1) - 400 * (x[i+1] - x[i]**2) * x[i] + 200 * (x[i] - x[i-1]**2)
    grad[N-1] = 200 * (x[N-1] - x[N-2]**2)
    return grad
def rosen_chained(x, alpha=1e2):
    x = [x] if isscalar(x[0]) else x  # scalar into list
    f = [(1. - x[0])**2 + sum(alpha * (x[:-1]**2 - x[1:])**2) for x in x]
    return f if len(f) > 1 else f[0]  # 1-element-list into scalar

def diffpow(x, rot=0):
    """Diffpow test objective function"""
    N = len(x)
    if rot:
        x = rotate(x)
    return sum(np.abs(x)**(2. + 4.*np.arange(N) / (N - 1.)))**0.5

def rosenelli(x):
    N = len(x)
    Nhalf = int((N + 1) / 2)
    return rosen(x[:Nhalf]) + elli(x[Nhalf:], cond=1)
def ridge(x, expo=2):
    x = [x] if isscalar(x[0]) else x  # scalar into list
    f = [x[0] + 100 * np.sum(x[1:]**2)**(expo / 2.) for x in x]
    return f if len(f) > 1 else f[0]  # 1-element-list into scalar
def ridgecircle(x, expo=0.5):
    """happy cat by HG Beyer"""
    a = len(x)
    s = sum(x**2)
    return ((s - a)**2)**(expo / 2) + s / a + sum(x) / a
def happycat(x, alpha=1. / 8):
    s = sum(x**2)
    return ((s - len(x))**2)**alpha + (s / 2 + sum(x)) / len(x) + 0.5
def flat(x):
    return 1
    return 1 if np.random.rand(1) < 0.9 else 1.1
    return np.random.randint(1, 30)
def ackley(x):
    """default domain is ``[-32.768, 32.768]^n``"""
    x = np.asarray(x)
    a, b = 20, 0.2
    s = np.exp(-b * np.sqrt(np.mean(x**2)))
    scos = np.exp(np.mean(np.cos(2 * np.pi * x)))
    return a * (1 - s) + (np.exp(1) - scos)
def branin(x):
    # in [0,15]**2
    y = x[1]
    x = x[0] + 5
    return (y - 5.1 * x**2 / 4 / np.pi**2 + 5 * x / np.pi - 6)**2 + 10 * (1 - 1 / 8 / np.pi) * np.cos(x) + 10 - 0.397887357729738160000
def goldsteinprice(x):
    x1 = x[0]
    x2 = x[1]
    return (1 + (x1 + x2 + 1)**2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)) * (
            30 + (2 * x1 - 3 * x2)**2 * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2)) - 3
def griewank(x):
    # was in [-600 600]
    x = (600. / 5) * x
    return 1 - np.prod(np.cos(x / np.sqrt(1. + np.arange(len(x))))) + sum(x**2) / 4e3
def rastrigin(x):
    """Rastrigin test objective function"""
    if not isscalar(x[0]):
        N = len(x[0])
        return [10 * N + sum(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x]
        # return 10*N + sum(x**2 - 10*np.cos(2*np.pi*x), axis=1)
    N = len(x)
    return 10 * N + sum(x**2 - 10 * np.cos(2 * np.pi * x))
def schaffer(x):
    """ Schaffer function x0 in [-100..100]"""
    N = len(x)
    s = x[0:N - 1]**2 + x[1:N]**2
    return sum(s**0.25 * (np.sin(50 * s**0.1)**2 + 1))

def schwefelelli(x):
    s = 0
    f = 0
    for i in _rglen(x):
        s += x[i]
        f += s**2
    return f
def schwefelmult(x, pen_fac=1e4):
    """multimodal Schwefel function with domain -500..500"""
    y = [x] if isscalar(x[0]) else x
    N = len(y[0])
    f = array([418.9829 * N - 1.27275661e-5 * N - sum(x * np.sin(np.abs(x)**0.5))
            + pen_fac * sum((abs(x) > 500) * (abs(x) - 500)**2) for x in y])
    return f if len(f) > 1 else f[0]
def schwefel2_22(x):
    """Schwefel 2.22 function"""
    return sum(np.abs(x)) + np.prod(np.abs(x))
def optprob(x):
    n = np.arange(len(x)) + 1
    f = n * x * (1 - x)**(n - 1)
    return sum(1 - f)
def lincon(x, theta=0.01):
    """ridge like linear function with one linear constraint"""
    if x[0] < 0:
        return np.NaN
    return theta * x[1] + x[0]
def rosen_nesterov(x, rho=100):
    """needs exponential number of steps in a non-increasing
    f-sequence.

    x_0 = (-1,1,...,1)
    See Jarre (2011) "On Nesterov's Smooth Chebyshev-Rosenbrock
    Function"

    """
    f = 0.25 * (x[0] - 1)**2
    f += rho * sum((x[1:] - 2 * x[:-1]**2 + 1)**2)
    return f
def powel_singular(x):
    # ((8 * np.sin(7 * (x[i] - 0.9)**2)**2 ) + (6 * np.sin()))
    res = np.sum((x[i - 1] + 10 * x[i])**2 + 5 * (x[i + 1] - x[i + 2])**2 +
                 (x[i] - 2 * x[i + 1])**4 + 10 * (x[i - 1] - x[i + 2])**4
                 for i in range(1, len(x) - 2))
    return 1 + res
def styblinski_tang(x):
    """in [-5, 5]
    found also in Lazar and Jarre 2016, optimum in f(-2.903534...)=0
    """
    # x_opt = N * [-2.90353402], seems to have essentially
    # (only) 2**N local optima
    return (39.1661657037714171054273576010019 * len(x))**1 + \
           sum(x**4 - 16 * x**2 + 5 * x) / 2

def trid(x):
    return sum((x-1)**2) - sum(x[:-1] * x[1:])

def bukin(x):
    """Bukin function from Wikipedia, generalized simplistically from 2-D.

    http://en.wikipedia.org/wiki/Test_functions_for_optimization"""
    s = 0
    for k in range((1+len(x)) // 2):
        z = x[2 * k]
        y = x[min((2*k + 1, len(x)-1))]
        s += 100 * np.abs(y - 0.01 * z**2)**0.5 + 0.01 * np.abs(z + 10)
    return s

def zerosum(x):
    """abs(sum xi) has an n-1 dimensionsal solution space.

    http://infinity77.net/global_optimization/test_functions_nd_Z.html#go_benchmark.ZeroSum
    https://github.com/CMA-ES/c-cmaes/issues/19#issue-323799789
    """
    s = np.sum(x)
    return 1 + 1e2 * np.abs(s)**0.5 if s else 0.0