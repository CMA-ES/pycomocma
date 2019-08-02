# -*- coding: utf-8 -*-
""""""
from __future__ import division, print_function
import warnings
import os
from collections import defaultdict, namedtuple
from ast import literal_eval
import numpy as np  # to replace 'inf' with np.inf and vice versa

class CompactGA(object):
    """Randomized search on the binary domain {0,1}^n.

    cGA samples two different solutions per iteration.

    Minimal example::

        dimension = 20
        opt = CompactGA(dimension * [0.5])  # initialise in domain middle
        while opt.best.f > 0 and not opt.stop:
            opt.iterate(sum)
        print("%d: fbest=%f" % (opt.evaluation, opt.best.f))

    finds argmin_{x in {0, 1}**20} sum(x) in ``opt.result``.

    Reference: Harik et al 1999.
    """

    def __init__(self, mean_):
        """takes as input the initial vector of probabilities to sample 1 vs 0.

        The vector length defines the dimension. Initial values of the
        probability vector, which is also the distribution mean, are
        usually chosen to be 0.5.
        """
        self.mean = np.asarray(mean_)
        self.dimension = len(self.mean)
        self.eta = 1 / self.dimension  # 1 / dim according to Harik et al 1999
        self.lower_p = 1 / self.dimension / 2
        """lower and 1 - upper bound for mean"""
        # bookkeeping
        self.best = BestSolution()
        self.fcurrent = np.nan
        self.evaluation = 0

    def iterate(self, f):
        """one iteration with ``f`` as fitness function to be minimized
        """
        x1 = np.array(np.random.rand(self.dimension) < self.mean, np.int8)
        x2 = x1
        while all(x2 == x1):
            x2 = np.array(np.random.rand(self.dimension) <= self.mean, np.int8)
        f1, f2 = f(x1), f(x2)
        self.evaluation += 2
        if f2 < f1:  # swap such that f1 is better
            f1, f2 = f2, f1
            x1, x2 = x2, x1
        if f1 < f2:  # update mean
            self.mean += self.eta * (x1 - x2)
            self.mean[self.mean < self.lower_p] = self.lower_p
            self.mean[self.mean > 1 - self.lower_p] = 1 - self.lower_p
        # bookkeeping
        self.fcurrent = f1
        self.best.update(f1, x1, self.evaluation)

    @property
    def stop(self):
        """dictionary containing satisfied termination conditions
        """
        stop_dict = {}
        self.stop_stagnation_evals = 2000 * self.dimension
        if self.evaluation > self.best.evaluation + self.stop_stagnation_evals:
            stop_dict['stagnation'] = self.stop_stagnation_evals
        return stop_dict

    @property
    def result(self):
        """for the time being `result` is the best seen solution
        """
        return self.best.x

class BestSolution(object):
    """Helper class to keep track of the best solution (minimization).

    All is stored in attributes ``x, f, evaluation``. The only reason
    for this class is to prevent code duplication of the `update`
    method.
    """
    def __init__(self):
        self.x = None
        self.f = np.inf
        self.evaluation = 0
    def update(self, f, x, evaluation):
        """update attributes ``f, x, evaluation`` if ``f < self.f``"""
        if f < self.f:
            self.f = f
            self.x = x[:]
            self.evaluation = evaluation

class BlankClass(object):
    """a blank container for (Matlab-like) out-of-the-box attribute access
    """

class ClassFromDict(object):
    """set class attributes from a `dict`"""
    def __init__(self, dict_):
        self._dict = dict(dict_)
        for key in dict_:
            setattr(self, key, dict_[key])
    @property
    def as_dict(self):
        """collect only original attributes, use ``__dict__`` to get also
        the attributes later added.
        """
        return dict([key, getattr(self, key)] for key in self._dict)

"""
        self.lower_p = 1 / self.dimension / 2
        self.lower_p = (1 / self.dimension) / 2
        self.lower_p = 1 / (self.dimension * 2)
        self.lower_p = 1 / (2 * self.dimension)
"""
def down_sample(x, y=None, len_=500):
    """return (index, x_down) if y is None, else (x_down, y_down).

    Example: ``plot(*down_sample(mydata))``

    """
    if y is None:
        x, y = np.arange(len(x)), np.asarray(x)
    while len(y) > 2 * len_:
        x = x[::2]
        y = y[::2]
    return x, y

def step_data(x, *args, **kwargs):
    """return x, y for `pyplot.step` ECDF of input `x`-data"""
    max_x = x[-1] + min((0.01 * (x[-1] - x[0]),
                         0.03 * (x[-1] - x[-2])))
    x = np.concatenate([x, [max_x]])
    y = np.linspace(0, 1, len(x), endpoint=True)
    return x, y

def sp1(a):
    """return the "average" of data in `a` under the assumption that
    non-finite entries contribute the same as the average finite entry
    and resampling is applied to get a finite entry.

    This measure has been called Q-measure or success performance one. It
    is computed as the sum of all finite entries devid
    """
    a = np.asarray(a)
    idx = np.isfinite(a)
    return len(a) * np.mean(a[idx]) / sum(idx) if sum(idx) else np.inf

class Results(object):
    """a container to communicate (changing) results via disk and with
    backups (of all data) in each step.

    The ``data`` attribute is a dictionary which contains all "results"
    data. Any new container reads previously saved data (with the same
    name) on initialization.

    Use cases::

        # save my_data_dict to file 'sweep_data.pydata'
        # thereby appending/updating existing data
        Results('sweep_data').save(my_data_dict)
        # load these (combined) data back
        my_data_dict = Results('sweep_data').load().data

        # save data after each trial
        res = Results('sweep_data')  # appends/merges data in case
        for dim in dimensions:
            for ...:
                ret = fmin(...)
                res.data[dim][...] = ret.evals if ret.success else np.inf
                res.save()  # like this we can never lose data


    Details: Saving/loading of nonfinite values with `ast.literat_eval` is
    covered with the ``value_string_mappings`` attribute.

    """
    def __init__(self,
                 name=None,
                 values_to_string = (np.inf, np.nan),
                verbose=1):
        self.filename = 'results_dict' if name is None else name
        if '.' not in self.filename[-8:]:
            self.filename = self.filename + '.pydict'
        path, name = os.path.split(self.filename)
        name, ext = os.path.splitext(name)
        self.backup_filename = os.path.join(path, '._' + name + '-backup' + ext)
        self._values_to_string_vals = values_to_string
        self.data = DataDict()  # {}
        self.meta_data = dict(info="", )
        try:
            self.load()
        except IOError:
            pass
        else:
            if self.data and verbose > 0:
                print('Results.__init__ loaded %d data key entries (%s ... %s)' % (
                      len(self.data),
                      str(sorted(self.data)[0]),
                      str(sorted(self.data)[-1])
                      ))

    def load(self):
        """TODO: this should optionally append to current data?"""
        if self.data:
            self.backup()
        with open(self.filename, 'rt') as f:
            try:
                self.data = DataDict(literal_eval(f.read()))
            except:
                print("""
    Please check whether inf or nan were used as simple value rather than
    in a sequence as [inf] or [nan].""")
                raise
        self._values_to_string(inverse=True)
        self.meta_data = self.data.pop('meta_data', None)
        return self

    def update(self, filename, backup=True):
        """Append data from ``filename`` to ``self.data``.
        
        To update self with a ``data`` `dict` instead a file, call
        ``self.data.update(data)``.
        """
        if backup:
            self.backup()
        if filename is not None:
            data = Results(filename).data
            for key in data:
                self.data[key] += data[key]
        return self

    def save(self, backup=True):
        """update `self.data` with `data` and save to disk"""
        if backup:
            try:
                with open(self.filename, 'rt') as f:
                    self.backup(f.read())
            except IOError: pass
        self._values_to_string()
        if self.meta_data is not None:
            self.data['meta_data'] = self.meta_data
        with open(self.filename, 'wt') as f:
            f.write(repr(self.data))
        self._values_to_string(inverse=True)
        return self  # necessary/useful?

    def _value_to_string(self, val, inverse=False):
        """return "correct" value"""
        if inverse:
            sval = val
            for val in self._values_to_string_vals:
                if repr(val) == sval:
                    return val
            return sval
        if val in self._values_to_string_vals:
            return repr(val)
        return val
    def _values_to_string(self, inverse=False):
        """replace values with strings or vice versa.

        Values to be replaced are defined at instance creation.
        """

        for a in self._arrays():
            try: len(a), a[0]
            except (TypeError, IndexError):
                try:
                    literal_eval(repr(a))
                except:
                    print("""
    Warning: non-sequence found as data value, which will fail when
             reading back serialized data. A simple fix is to use
             ``[val]`` instead of ``val`` or avoid values which
             `ast.literal_eval` cannot digest, like ``inf``, ``nan``, etc.
             """)
            else:
                for i in range(len(a)):
                    a[i] = self._value_to_string(a[i], inverse)

        for val in self._values_to_string_vals:
            new, old = repr(val), val
            if inverse:
                new, old = old, new
            if old in self.data:
                self.data[new] = self.data[old]
                del self.data[old]

    def _arrays(self, data=None):
        """return a flat list of (references to) all non-dicts in data.

        Traverses recursively down into dictionaries.
        """
        if data is None:
            data = self.data
        if isinstance(data, dict):
            res = []
            for key in data:  # travers down
                if key != 'meta_data':
                    res += self._arrays(data[key])
            return res
        else:  # return leave
            return [data]

    def backup(self, data=None):
        """append data to backup file"""
        self._values_to_string()
        if data is None:
            data = self.data
        with open(self.backup_filename, 'at') as f:
            f.write(repr(data))
            f.write('\n')
        self._values_to_string(inverse=True)

class DataDict(defaultdict):
    """A dictionary with a parameter value, e.g. dimension, as keys and
    a list/sequence of results, e.g. runtimes, as value for each key.

    This class provides simple computations on this kind of data,
    like ``x, y = .xy_arrays() == sorted(keys), sp1(values)``.

    A main functionality is the method `clean`, which joins all entries
    which have almost equal keys. This allows to have a `float` parameter
    as key.

    If the dictionary values are not lists, one may get rather unexpected
    results or exceptions.

    Details: this class allows to use `float` values as keys when
    `clean_key` and `set_clean` are used to access the data in the
    `dict`. Inheriting from `defaultdict` with ``[]`` as default value,
    the syntax::

        data = DataDict()
        data[first_key] += [first_data_point]

    without initialization of the key value works perfectly fine.

    Caveat: small values are considered as the same key, even if they are
    close to zero. Either use a different comparison via the `equal`
    keyword parameter, or use ``1 / key_value`` or `log(key_value)``.

    TODO: consider `numpy.allclose` for almost equal comparison
    """
    def __init__(self, dict_=None):
        """Use ``dict(dict_.data or dict_)``, and `dict_.meta_data` for
        initialization.

        Details: `dict_.meta_data` are not copied.
        """
        defaultdict.__init__(self, lambda: [])
        if dict_ is not None:
            if hasattr(dict_, 'meta_data'):
                self.meta_data = dict_.meta_data
            if hasattr(dict_, 'data'):
                self.update(dict_.data)
            else:
                self.update(dict_)
    def xy_arrays(self, agg=sp1):
        """return an array of sorted keys and an array of the respectively
        aggregated values of this `dict`.

        For example to be used as
        ``plot(*Sweep('data-sweep-dim').sorted_arrays())``.

        Parameter `agg` determines the function to be used to aggregate
        data values, by default `sp1`

        """
        keys = [k for k in sorted(self) if np.isscalar(k)]
        return (np.asarray(keys),
                np.asarray([agg(self[k]) for k in keys]))

    def argmin(self, agg=sp1, slack=1.0, slack_index_shift=+1):
        x, y = self.xy_arrays(agg)
        idxmin = np.argmin(y)
        if not slack_index_shift:
            return x[idxmin]
        i = idxmin + slack_index_shift
        while i >= 0 and len(y) > i and y[i] <= slack * y[idxmin]:
            i += slack_index_shift
        return x[i - slack_index_shift]


    def clean(self, equal=lambda x, y: x - 1e-6 < y < x + 1e-6):
        """merge keys which have almost the same value"""
        for key in list(self.keys()):
            if key not in self:  # self.keys() changes in the process
                continue
            self.clean_key(key, equal=equal)
        return self

    def clean_key(self, key, equal=lambda x, y: x - 1e-6 < y < x + 1e-6):
        """set similar key values all to be `key`, return `key`.

        Use method `set_clean` to access and change the clean-key
        dictionary *value* more conveniently.
        """
        meta_data = self.pop('meta_data', None)
        while self._near_key(key, equal) is not key:
            k = self._near_key(key, equal)
            assert k != key
            self[key] += self[k]  # def join_(a, b): return list(a) + list(b) could become input
            del self[k]  # del a superfluous reference
        if meta_data is not None:
            self['meta_data'] = meta_data
        return key

    def get_near(self, key, equal=lambda x, y: x - 1e-6 < y < x + 1e-6):
        """get the merged values list of all nearby keys.

        Caveat: the returned value is a new list

        :See also: `clean`, `set_clean`.

        """
        res = [] if key not in self else list(self[key])  # prevent adding the key
        done_keys = [key]
        while self._near_key(key, equal, done_keys) is not key:
            k = self._near_key(key, equal, done_keys)
            done_keys += [k]
            res += self[k]
        return res

    def set_clean(self, key):
        """join all entries with similar `key` and return the new value,
        a joined list of all respective values.

        Example::

            data.set_clean(key) += [new_data_point]

            # same as
            data[data.clean_key(key)] += [new_data_point]

            # similar as
            data[key] += [new_data_point]
            data.clean()  # cleans *all* keys

        """
        self.clean_key(key)
        return self[key]  # same as self[self.clean_key(key)]

    def _near_key(self, key, equal=lambda x, y: x - 1e-6 < y < x + 1e-6,
                  exclude=None):
        """return a key in self which is ``equal`` to ``key`` and otherwise ``key``.
        """
        if exclude is None:
            exclude = []
        for k in sorted(self.keys()):  # sorted doesn't work with float and str
            try:  # prevent type error from equal(float, str)
                if equal(k, key) and k != key and k not in exclude:
                    return k
            except TypeError:
                pass
        return key
    @property
    def successes(self):
        """return a class instance with attributes `x` (i.e. keys), `n`,
        `nsucc`, and `rate`.
        """
        keys = [k for k in sorted(self) if np.isscalar(k)]
        nsucc = np.asarray([sum(np.isfinite(self[k])) for k in keys])
        n = np.asarray([len(self[k]) for k in keys])
        try:
            Successes = namedtuple('Successes',
                                   ['x', 'nsucc', 'n', 'rate'])
            res = Successes(np.asarray(keys), nsucc, n, nsucc / n)
        except:
            res = ClassFromDict(
                {'x': np.asarray(keys),
                 'nsucc': nsucc,
                 'n': n,
                 'rate': nsucc / n,
                 })
        return res

    def percentile(self, prctile, agg=sp1, samples=100):
        """percentile based on bootstrapping"""
        raise NotImplementedError
        keys = [k for k in sorted(self) if np.isscalar(k)]
        res = []
        for k in keys:
            bstrapped = []
            for i in range(samples):
                idx = np.random.randint(len(self[k]))
                data = self[k][idx]
                bstrapped.append(agg(data))
            res.append(np.percentile(bstrapped, prctile))
        return np.asarray(keys), np.asarray(res)

    def __repr__(self):
        return repr(dict(self))

del division, print_function