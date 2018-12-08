# -*- coding: utf-8 -*-
"""This module contains, for the time being, a single MOO archive class.

A bi-objective nondominated archive as sorted list with incremental
update in logarithmic time.

"""
from __future__ import division, print_function, unicode_literals
__author__ = "Nikolaus Hansen and ..."
__license__ = "BSD 3-clause"
__version__ = "0.5.0"
del division, print_function, unicode_literals

# from collections import deque  # does not support deletion of slices!?
import bisect as _bisect # to find the insertion index efficiently
import fractions
from math import inf
import warnings as _warnings

class BiobjectiveNondominatedSortedList(list):
    """A sorted list of non-dominated unique objective-pairs.

    Non-domination here means smaller in at least one objective. The list is
    sorted (naturally) by the first objective. No equal entries in either
    objective exist in the list (assuming it is in a consistent state).

    The operation

    >>> from moarchiving import BiobjectiveNondominatedSortedList
    >>> any_list = BiobjectiveNondominatedSortedList(any_list)  # doctest:+SKIP

    sorts and prunes the pair list `any_list` to become a consistent
    nondominated sorted archive.

    Afterwards, the methods `add` and `add_list` keep the list always
    in a consistent state. If a reference point was given on initialization,
    also the hypervolume of the archive is computed and updated.
    
    Removing elements with `pop` or `del` keeps the archive sorted and
    non-dominated but does not update the hypervolume, which hence
    becomes inconsistent.

    >>> a = BiobjectiveNondominatedSortedList([[1,0.9], [0,1], [0,2]])
    >>> a
    [[0, 1], [1, 0.9]]
    >>> a.add([0, 1])  # doesn't change anything, [0, 1] is not duplicated
    >>> BiobjectiveNondominatedSortedList(
    ...     [[-0.749, -1.188], [-0.557, 1.1076],
    ...     [0.2454, 0.4724], [-1.146, -0.110]])
    [[-1.146, -0.11], [-0.749, -1.188]]
    >>> a._asserts()  # consistency assertions

    Details: This list doesn't prevent the user to insert a new element
    anywhere and hence get into an inconsistent state. Inheriting from
    `sortedcontainers.SortedList` would ensure that the `list` remains
    at least sorted.

    See also:
    https://pypi.org/project/sortedcontainers
    https://code.activestate.com/recipes/577197-sortedcollection/
    https://pythontips.com/2016/04/24/python-sorted-collections/

    DONE: implement large-precision hypervolume computation.
    DONE (method remove): implement a `delete` method that also updates the hypervolume.
    TODO (DONE): implement a copy method
    TODO: currently, points beyond the reference point (which do not contribute
    to the hypervolume) are discarded. We may want to keep them, for simplicity
    in a separate list?

    """
    # Default Values for respective instance attributes
    make_expensive_asserts = False
    hypervolume_final_float_type = fractions.Fraction  # HV computation takes three times longer, precision may be more relevant here
    hypervolume_computation_float_type = fractions.Fraction  # HV computation takes three times longer, precision may be less relevant here
    # hypervolume_final_float_type = float  # lambda x: x is marginally faster
    # hypervolume_computation_float_type = float  # may be a good compromise
    maintain_contributing_hypervolumes = False

    def __init__(self,
                 list_of_f_pairs=None,
                 reference_point=None,
                 sort=sorted):
        """`list_of_f_pairs` does not need to be sorted.

        f-pairs beyond the `reference_point` are pruned away. The
        `reference_point` is also used to compute the hypervolume.

        ``sort=lambda x: x`` will prevent a sort, which
        can be useful if the list is already sorted.

        CAVEAT: the interface, in particular the positional interface
        may change in future versions.
        """
        self.make_expensive_asserts = BiobjectiveNondominatedSortedList.make_expensive_asserts
        self.hypervolume_final_float_type = BiobjectiveNondominatedSortedList.hypervolume_final_float_type
        self.hypervolume_computation_float_type = BiobjectiveNondominatedSortedList.hypervolume_computation_float_type
        self.maintain_contributing_hypervolumes = BiobjectiveNondominatedSortedList.maintain_contributing_hypervolumes

        if list_of_f_pairs is not None and len(list_of_f_pairs):
            try:
                list_of_f_pairs = list_of_f_pairs.tolist()
            except:
                pass
            if len(list_of_f_pairs[0]) != 2:
                raise ValueError("need elements of len 2, got %s"
                                 " as first element" % str(list_of_f_pairs[0]))
            list.__init__(self, sort(list_of_f_pairs))
            # super(BiobjectiveNondominatedSortedList, self).__init__(sort(list_of_f_pairs))
        if reference_point is not None:
            self.reference_point = list(reference_point)
        else:
            self.reference_point = reference_point
        self.prune()  # remove dominated entries, uses in_domain, hence ref-point
        if self.maintain_contributing_hypervolumes:
            self._contributing_hypervolumes = self.contributing_hypervolumes
            raise NotImplementedError('update of _contributing_hypervolumes in _add_HV and _subtract_HV not implemented')
        else:
            self._contributing_hypervolumes = []
        self._set_HV()
        self.make_expensive_asserts and self._asserts()

    def add(self, f_pair):
        """insert `f_pair` in `self` if it is not (weakly) dominated.

        Return index at which the insertion took place or `None`. The
        list remains sorted in the process.

        The list remains non-dominated with unique elements, which
        means that some or many or even all of its present elements may
        be removed.

        Implementation detail: For performance reasons, `insert` is
        avoided in favor of `__setitem__`, if possible.
        """
        f_pair = list(f_pair)  # convert array to list
        if len(f_pair) != 2:
            raise ValueError("argument `f_pair` must be of length 2, was"
                             " ``%s``" % str(f_pair))
        if not self.in_domain(f_pair):
            self._removed = [f_pair]
            return None
        idx = self.bisect_left(f_pair)
        if self.dominates_with(idx - 1, f_pair) or self.dominates_with(idx, f_pair):
            if f_pair not in self[idx - 1:idx + 1]:
                self._removed = [f_pair]
            return None
        assert idx == len(self) or not f_pair == self[idx]
        # here f_pair now is non-dominated
        self._add_at(idx, f_pair)
        # self.make_expensive_asserts and self._asserts()
        return idx

    def _add_at(self, idx, f_pair):
        """add `f_pair` at position `idx` and remove dominated elements.

        This method assumes that `f_pair` is not weakly dominated by
        `self` and that `idx` is the correct insertion place e.g.
        acquired by `bisect_left`.
        """
        if idx == len(self) or f_pair[1] > self[idx][1]:
            self.insert(idx, f_pair)
            self._add_HV(idx)
            # self.make_expensive_asserts and self._asserts()
            return
        # here f_pair now dominates self[idx]
        idx2 = idx + 1
        while idx2 < len(self) and f_pair[1] <= self[idx2][1]:
            # f_pair also dominates self[idx2]
            # self.pop(idx)  # slow
            # del self[idx]  # slow
            idx2 += 1  # delete later in a chunk
        self._subtract_HV(idx, idx2)
        self[idx] = f_pair  # on long lists [.] is much cheaper than insert
        self._removed = self[idx + 1:idx2]
        del self[idx + 1:idx2]  # can make `add` 20x faster
        self._add_HV(idx)
        assert len(self) >= 1
        # self.make_expensive_asserts and self._asserts()

    def remove(self, f_pair):
        """remove element `f_pair`.

        Raises a `ValueError` (like `list`) if ``f_pair is not in self``.
        To avoid the error, checking ``if f_pair is in self`` first is a
        possible coding solution, like

        >>> from moarchiving import BiobjectiveNondominatedSortedList
        >>> nda = BiobjectiveNondominatedSortedList([[2, 3]])
        >>> f_pair = [1, 2]
        >>> assert [2, 3] in nda and f_pair not in nda
        >>> if f_pair in nda:
        ...     nda.remove(f_pair)

        """
        if not hasattr(self, '_remove_test_warning'):
            _warnings.warn("BiobjectiveNondominatedSortedList.remove has never been tested")
            self._remove_test_warning = True
        idx = self.index(f_pair)
        self._subtract_HV(idx)
        self._removed = [self[idx]]
        del self[idx]  # == list.remove(self, f_pair)

    def add_list(self, list_of_f_pairs):
        """insert a list of f-pairs which doesn't need to be sorted.

        This is just a shortcut for looping over `add`, but `discarded`
        now contains the discarded elements from all `add` operations.

        >>> from moarchiving import BiobjectiveNondominatedSortedList
        >>> arch = BiobjectiveNondominatedSortedList()
        >>> list_of_f_pairs = [[1, 2], [0, 3]]
        >>> for f_pair in list_of_f_pairs:
        ...     arch.add(f_pair)  # return insert index or None
        0
        0
        >>> arch == sorted(list_of_f_pairs)  # both entries are nondominated
        True
        >>> arch.compute_hypervolume([3, 4]) == 5.0
        True

        Return `None`.

        Details: discarded does not contain elements of `list_of_f_pairs`.
        When `list_of_pairs` is already sorted, `merge` may have
        a small performance benefit.
        """
        nb = len(self)
        removed = []
        # should we better create a non-dominated list and do a merge?
        for f_pair in list_of_f_pairs:
            if self.add(f_pair) is not None:
                removed += [self._removed]  # slightly faster than .extend
        self._removed = removed  # could contain elements of `list_of_f_pairs`
        self.make_expensive_asserts and self._asserts()

    def merge(self, list_of_f_pairs):
        """merge in a sorted list of f-pairs.

        The list can contain dominated pairs, which are discarded during
        the merge.

        Return `None`.

        Details: merging 200 into 100_000 takes 3e-4s vs 4e-4s with
        `add_list`. The `discarded` property is not consistent with the
        overall merge.
        """
        # _warnings.warn("merge was never thoroughly tested, use `add_list`")
        for f_pair in list_of_f_pairs:
            if not self.in_domain(f_pair):
                continue
            f_pair = list(f_pair)  # convert array to list
            idx = self.bisect_left(f_pair, idx)
            if self.dominates_with(idx - 1, f_pair) or self.dominates_with(idx, f_pair):
                continue
            self._add_at(idx, f_pair)
        self.make_expensive_asserts and self._asserts()

    def copy(self):
        """return a "deep" copy of `self`"""
        _warnings.warn('BiobjectiveNondominatedSortedList.copy has never been tested')
        nda = BiobjectiveNondominatedSortedList()
        for d in self.__dict__:
            nda[d] = self[d]
        # now fix all mutable references as a true copy
        list.__init__(nda, self)
        nda.reference_point = [xi for xi in self.reference_point]
        nda._hypervolume = self.hypervolume_final_float_type(self._hypervolume)  # with Fraction not necessary
        nda._contributing_hypervolumes = [hv for hv in self._contributing_hypervolumes]
        return nda

    def bisect_left(self, f_pair, lowest_index=0):
        """return index where `f_pair` may need to be inserted.

        Smaller indices have a strictly better f1 value or they have
        equal f1 and better f2 value.

        `lowest_index` restricts the search from below.

        Details: This method does a binary search in `self` using
        `bisect.bisect_left`.
        """
        return _bisect.bisect_left(self, f_pair, lowest_index)

    def dominates(self, f_pair):
        """return `True` if any element of `self` dominates or is equal to `f_pair`.

        Otherwise return `False`.

        >>> from moarchiving import BiobjectiveNondominatedSortedList as NDA
        >>> a = NDA([[0.39, 0.075], [0.0087, 0.14]])
        >>> a.dominates(a[0])  # is always True if `a` is not empty
        True
        >>> a.dominates([-1, 33]) or a.dominates([33, -1])
        False
        >>> a._asserts()

        See also `bisect_left` to find the closest index.
        """
        idx = self.bisect_left(f_pair)
        if self.dominates_with(idx - 1, f_pair) or self.dominates_with(idx, f_pair):
            return True
        return False

    def dominates_with(self, idx, f_pair):
        """return `True` if ``self[idx]`` dominates or is equal to `f_pair`.

        Otherwise return `False` or `None` if `idx` is out-of-range.

        >>> from moarchiving import BiobjectiveNondominatedSortedList as NDA
        >>> NDA().dominates_with(0, [1, 2]) is None  # empty NDA
        True

        """
        if idx < 0 or idx >= len(self):
            return None
        if self[idx][0] <= f_pair[0] and self[idx][1] <= f_pair[1]:
            return True
        return False

    def dominators(self, f_pair, number_only=False):
        """return the list of all `f_pair`-dominating elements in `self`,

        including an equal element. ``len(....dominators(...))`` is
        hence the number of dominating elements which can also be obtained
        without creating the list with ``number_only=True``.

        >>> from moarchiving import BiobjectiveNondominatedSortedList as NDA
        >>> a = NDA([[1.2, 0.1], [0.5, 1]])
        >>> len(a)
        2
        >>> a.dominators([2, 3]) == a
        True
        >>> a.dominators([0.5, 1])
        [[0.5, 1]]
        >>> len(a.dominators([0.6, 3])), a.dominators([0.6, 3], number_only=True)
        (1, 1)
        >>> a.dominators([0.5, 0.9])
        []

        """
        idx = self.bisect_left(f_pair)
        if idx < len(self) and self[idx] == f_pair:
            res = 1 if number_only else [self[idx]]
        else:
            res = 0 if number_only else []
        idx -= 1
        while idx >= 0 and self[idx][1] <= f_pair[1]:
            if number_only:
                res += 1
            else:
                res.insert(0, self[idx])  # keep sorted
            idx -= 1
        return res

    def in_domain(self, f_pair, reference_point=None):
        """return `True` if `f_pair` is dominating the reference point,

        `False` otherwise. `True` means that `f_pair` contributes to
        the hypervolume if not dominated by other elements.

        `f_pair` may also be an index in `self` in which case
        ``self[f_pair]`` is tested to be in-domain.

        >>> from moarchiving import BiobjectiveNondominatedSortedList as NDA
        >>> a = NDA([[2.2, 0.1], [0.5, 1]], reference_point=[2, 2])
        >>> assert len(a) == 1
        >>> a.in_domain([0, 0])
        True
        >>> a.in_domain([2, 1])
        False
        >>> all(a.in_domain(ai) for ai in a)
        True
        >>> a.in_domain(0)
        True

        TODO: improve name?
        """
        if reference_point is None:
            reference_point = self.reference_point
        if reference_point is None:
            return True
        try:
            f_pair = self[f_pair]
        except TypeError:
            pass
        except IndexError:
            raise  # return None
        if (f_pair[0] >= reference_point[0] or
            f_pair[1] >= reference_point[1]):
            return False
        return True

    @property
    def hypervolume(self):
        """hypervolume of the entire list w.r.t. the "initial" reference point.

        Raise `ValueError` when no reference point was given initially.

        >>> from moarchiving import BiobjectiveNondominatedSortedList as NDA
        >>> a = NDA([[0.5, 0.4], [0.3, 0.7]], [2, 2.1])
        >>> a._asserts()
        >>> a.reference_point == [2, 2.1]
        True
        >>> abs(a.hypervolume - a.compute_hypervolume(a.reference_point)) < 1e-11
        True
        >>> a.add([0.2, 0.8])
        0
        >>> a._asserts()
        >>> abs(a.hypervolume - a.compute_hypervolume(a.reference_point)) < 1e-11
        True
        >>> a.add([0.3, 0.6])
        1
        >>> a._asserts()
        >>> abs(a.hypervolume - a.compute_hypervolume(a.reference_point)) < 1e-11
        True

        """
        if self.reference_point is None:
            raise ValueError("to compute the hypervolume a reference"
                             " point is needed (must be given initially)")
        if self.make_expensive_asserts:
            assert abs(self._hypervolume - self.compute_hypervolume(self.reference_point)) < 1e-12
        return self._hypervolume

    @property
    def contributing_hypervolumes(self):
        """`list` of contributing hypervolumes.

        Elements in the list are of type
        `self.hypervolume_computation_float_type`.
        Conversion to `float` in a list comprehension should always be
        possible.

        Changing this list will have unexpected consequences if
        `self.maintain_contributing_hypervolumes`,

        Details: The "initial" reference point is used for the outer
        points. If none is given, `inf` is used as reference.
        For the time being, the contributing hypervolumes are
        computed each time from scratch.

        :See also: `contributing_hypervolume`
        """
        if self.maintain_contributing_hypervolumes:
            if not hasattr(self, '_contributing_hypervolumes'):
                self._contributing_hypervolumes = [
                    self.contributing_hypervolume(i)
                    for i in range(len(self))]
            if len(self._contributing_hypervolumes) == len(self):
                return self._contributing_hypervolumes
            _warnings.warn("contributing hypervolumes seem not consistent")
        return [self.contributing_hypervolume(i)
                for i in range(len(self))]

    def contributing_hypervolume(self, idx):
        """return contributing hypervolume of element `idx`.

        The return type is ``self.hypervolume_computation_float_type` and
        by default `fractions.Fraction`, which can be converted to `float`
        like ``float(....contributing_hypervolume(idx))``.
        """
        if idx == 0:
            y = self.reference_point[1] if self.reference_point else inf
        else:
            y = self[idx - 1][1]
        if idx in (len(self) - 1, -1):
            x = self.reference_point[0] if self.reference_point else inf
        else:
            x = self[idx + 1][0]
        if inf in (x, y):
            return inf
        Fc = self.hypervolume_computation_float_type
        dHV = (Fc(x) - Fc(self[idx][0])) * (Fc(y) - Fc(self[idx][1]))
        assert dHV >= 0
        return dHV

    def distance_to_pareto_front(self, f_pair):
        """of a dominated `f_pair` without considering the reference domain.

        Non-dominated points have (by definition) a distance of zero.

        Details: the distance is computed by iterating over some kink
        points ``(self[i+1][0], self[i][1])``.
        """
        if len(self) == 0:  # f_pair is not dominated
            return 0  # return minimum distance

        # distances to the front boundary given by the extreme points:
        squared_distances = [max((0, f_pair[0] - self[0][0]))**2,
                             max((0, f_pair[1] - self[-1][1]))**2]
        if len(self) == 1:
            return min(squared_distances)**0.5
        for idx in range(self.bisect_left(f_pair), 0, -1):
            if idx == len(self):
                continue
            squared_distances.append(
                max((0, f_pair[1] - self[idx - 1][1]))**2 +
                max((0, f_pair[0] - self[idx][0]))**2)
            if self[idx][1] >= f_pair[1] or idx == 1:
                break
        if self.make_expensive_asserts:
            assert min(squared_distances[2:]) == min(
                        [max((0, f_pair[0] - self[i + 1][0]))**2 +
                         max((0, f_pair[1] - self[i][1]))**2
                         for i in range(len(self) - 1)])
        return min(squared_distances)**0.5

    def hypervolume_improvement(self, f_pair):
        """return how much `f_pair` would improve the hypervolumen.

        If dominated, return the distance to the empirical
        pareto front times -1.
        """
        if not hasattr(self, '_hypervolume_improvement_test_warning'):
            self._hypervolume_improvement_test_warning = True
            _warnings.warn("BiobjectiveNondominatedSortedList.hypervolume_"
                           "improvement has never been tested")

        penalty = max((0, f_pair[0] - self.reference_point[0]))**2 + \
                  max((0, f_pair[1] - self.reference_point[1]))**2 \
                  if self.reference_point else 0
        if self.dominates(f_pair):
            return -self.distance_to_pareto_front(f_pair) - penalty
        if not self.in_domain(f_pair):
            return -penalty
        hv0 = self.hypervolume
        removed = self.discarded  # to get back previous state
        self.add(f_pair)
        add_back = self.discarded
        hv1 = self.hypervolume
        self.remove(f_pair)
        if add_back:
            # print(add_back)
            self.add_list(add_back)
        self._removed = removed
        return self.hypervolume_computation_float_type(hv1) - hv0

    def _set_HV(self):
        """set current hypervolume value using `self.reference_point`.

        Raise `ValueError` if `self.reference_point` is `None`.

        TODO: we may need to store the list of _contributing_ hypervolumes
        to handle numerical rounding errors later.
        """
        if self.reference_point is None:
            return None
        self._hypervolume = self.compute_hypervolume(self.reference_point)
        return self._hypervolume

    def compute_hypervolume(self, reference_point):
        """return hypervolume w.r.t. `reference_point`"""
        if reference_point is None:
            raise ValueError("to compute the hypervolume a reference"
                             " point is needed (was `None`)")
        Fc = self.hypervolume_computation_float_type
        Ff = self.hypervolume_final_float_type
        hv = Ff(0.0)
        idx = 0
        while idx < len(self) and not self.in_domain(self[idx], reference_point):
            idx += 1
        if idx < len(self):
            hv += Ff((Fc(reference_point[0]) - Fc(self[idx][0])) * (Fc(reference_point[1]) - Fc(self[idx][1])))
            idx += 1
        while idx < len(self) and self.in_domain(self[idx], reference_point):
            hv += Ff((Fc(reference_point[0]) - Fc(self[idx][0])) * (Fc(self[idx - 1][1]) - Fc(self[idx][1])))
            idx += 1
        return hv

    def compute_hypervolumes(self, reference_point):
        """depricated, subject to removal, see `compute_hypervolume` and `contributing_hypervolumes`.

        Never implemented: return list of contributing hypervolumes w.r.t.
        reference_point
        """
        raise NotImplementedError()
        # construct self._hypervolumes_list
        # keep sum of different size elements separate,
        # say, a dict of index lists as indices[1e12] indices[1e6], indices[1], indices[1e-6]...
        hv = {}
        for key in indices:
            hv[key] = sum(_hypervolumes_list[i] for i in indices[key])
        # we may use decimal.Decimal to compute the sum of hv
        decimal.getcontext().prec = 88
        hv_sum = sum([decimal.Decimal(hv[key]) for key in hv])

    def _subtract_HV(self, idx0, idx1=None):
        """remove contributing hypervolumes of elements ``self[idx0] to self[idx1 - 1]``.

        TODO: also update list of contributing hypervolumes in case.
        """
        if self.maintain_contributing_hypervolumes:
            raise NotImplementedError("update list of hypervolumes")
            del self._contributing_hypervolumes[idx]
            # we also need to update the contributing HVs of the neighbors
        if self.reference_point is None:
            return None
        if idx1 is None:
            idx1 = idx0 + 1
        if idx0 == 0:
            y = self.reference_point[1]
        else:
            y = self[idx0 - 1][1]
        Fc = self.hypervolume_computation_float_type
        Ff = self.hypervolume_final_float_type
        dHV = Fc(0.0)
        for idx in range(idx0, idx1):
            if idx == len(self) - 1:
                assert idx < len(self)
                x = self.reference_point[0]
            else:
                x = self[idx + 1][0]
            dHV -= (Fc(x) - Fc(self[idx][0])) * (Fc(y) - Fc(self[idx][1]))
        assert dHV <= 0  # and without loss of precision strictly smaller
        if (Ff is not fractions.Fraction or not isinstance(self._hypervolume, Ff)) \
                and self._hypervolume and abs(dHV) / self._hypervolume < 1e-9:
            _warnings.warn("_subtract_HV: %f + %f loses many digits of precision"
                          % (dHV, self._hypervolume))
        self._hypervolume += Ff(dHV)
        if self._hypervolume < 0:
            _warnings.warn("adding %.16e to the hypervolume lead to a"
                          " negative hypervolume value of %.16e" %
                          (dHV, self._hypervolume))
        # assert self._hypervolume >= 0
        return dHV

    def _add_HV(self, idx):
        """add contributing hypervolume of ``self[idx]`` to hypervolume.

        TODO: also update list of contributing hypervolumes in case.
        """
        dHV = self.contributing_hypervolume(idx)
        if self.maintain_contributing_hypervolumes:
            raise NotImplementedError("update list of hypervolumes")
            self._contributing_hypervolumes.insert(idx, dHV)
            if idx > 0:
                self._contributing_hypervolumes[idx - 1] = self.contributing_hypervolume(idx - 1)
            if idx < len(self) - 1:
                self._contributing_hypervolumes[idx + 1] = self.contributing_hypervolume(idx + 1)
            # TODO: proof read
        if self.reference_point is None:
            return None
        Ff = self.hypervolume_final_float_type
        if self._hypervolume and (
                        Ff is not fractions.Fraction or not isinstance(self._hypervolume, Ff)) \
                and dHV / self._hypervolume < 1e-9:
            _warnings.warn("_subtract_HV: %f + %f loses many digits of precision"
                          % (dHV, self._hypervolume))
        self._hypervolume += Ff(dHV)
        return dHV

    def prune(self):
        """remove dominated or equal entries assuming that the list is sorted.

        Return number of dropped elements.

        Implementation details: pruning from right to left may be
        preferable, because list.insert(0) is O(n) while list.append is
        O(1), however it is not possible with the given sorting: in
        principle, the first element may dominate all others, which can
        only be discovered in the last step when traversing from right
        to left. This suggests that reverse sort may be better for
        pruning or we should inherit from `collections.deque` instead
        from `list`, but `deque` seems not to support deletion of slices.
        """
        nb = len(self)
        i = 1
        removed = []
        while i < len(self):
            i0 = i
            while i < len(self) and (self[i][1] >= self[i0 - 1][1] or
                                         not self.in_domain(self[i])):
                i += 1
                # self.pop(i + 1)  # about 10x slower in notebook test
            # prepare indices for the removed list
            i0r = i0
            if i0 > 0:
                while i0r < i:
                    if self[i0r] == self[i0 - 1]:
                        i0r += 1  # skip self[i0r] as removed because it is still in self
                    else:
                        break
            ir = i
            if i + 1 < len(self):
                while ir > i0r:
                    if self[ir] == self[i + 1]:
                        ir -= 1  # skip self[ir] as removed as it is in self
                    else:
                        break
            removed += self[i0r:ir]
            del self[i0:i]
            i = i0 + 1
        self._removed = removed  # [p for p in removed if p not in self]
        if self.maintain_contributing_hypervolumes:
            raise NotImplementedError
            self._contributing_hypervolumes = [  # simple solution
                self.contributing_hypervolume(i)
                for i in range(len(self))]
        return nb - len(self)

    @property
    def discarded(self):
        """`list` of f-pairs discarded in the last relevant method call.

        Methods covered are `__init__`, `prune`, `add`, and `add_list`.
        Removed duplicates are not element of the discarded list.
        When not inserted and not already in `self` also the input
        argument(s) show(s) up in `discarded`.

        Example to create a list of rank-k-non-dominated fronts:

        >>> from moarchiving import BiobjectiveNondominatedSortedList as NDA
        >>> all_ = [[0.1, 1], [-2, 3], [-4, 5], [-4, 5], [-4, 4.9]]
        >>> nda_list = [NDA(all_)]  # rank-0-non-dominated
        >>> while nda_list[-1].discarded:
        ...     nda_list += [NDA(nda_list[-1].discarded)]
        >>> assert [len(p) for p in nda_list] == [3, 1]

        """
        try:
            return self._removed
        except AttributeError:
            return []

    def _asserts(self):
        """make all kind of consistency assertions.

        >>> import moarchiving
        >>> a = moarchiving.BiobjectiveNondominatedSortedList(
        ...    [[-0.749, -1.188], [-0.557, 1.1076],
        ...    [0.2454, 0.4724], [-1.146, -0.110]], [10, 10])
        >>> a._asserts()
        >>> for i in range(len(a)):
        ...    assert a.contributing_hypervolume(i) == a.contributing_hypervolumes[i]
        >>> assert all(map(lambda x, y: x - 1e-9 < y < x + 1e-9,
        ...               a.contributing_hypervolumes,
        ...               [4.01367, 11.587422]))
        >>> for p in list(a):
        ...     a.remove(p)
        >>> assert len(a) == 0

        >>> from numpy.random import rand
        >>> a = moarchiving.BiobjectiveNondominatedSortedList([list(r) for r in rand(30, 2)],
        ...                                                   reference_point=[2, 2])
        >>> a.make_expensive_asserts = True
        >>> for f_pair in rand(30, 2):
        ...     hi = a.hypervolume_improvement(list(f_pair))


        """
        assert sorted(self) == self
        for pair in self:
            assert self.count(pair) == 1
        tmp = BiobjectiveNondominatedSortedList.make_expensive_asserts
        BiobjectiveNondominatedSortedList.make_expensive_asserts = False
        assert BiobjectiveNondominatedSortedList(self) == self
        BiobjectiveNondominatedSortedList.make_expensive_asserts = tmp
        for pair in self:
            assert self.dominates(pair)
            assert not self.dominates([v - 0.001 for v in pair])
        if self.reference_point is not None:
            assert abs(self._hypervolume - self.compute_hypervolume(self.reference_point)) < 1e-11
            assert sum(self.contributing_hypervolumes) < self.hypervolume + 1e-11
        if self.maintain_contributing_hypervolumes:
            assert len(self) == len(self._contributing_hypervolumes)
        assert len(self) == len(self.contributing_hypervolumes)
        # for i in range(len(self)):
        #     assert self.contributing_hypervolume(i) == self.contributing_hypervolumes[i]

        # asserts that use numpy for convenience
        try:
            import numpy as np
        except ImportError:
            _warnings.warn("asserts using numpy omitted")
        else:
            if len(self) > 1:
                diffs = np.diff(self, 1, 0)
                assert all(diffs[:, 0] > 0)
                assert all(diffs[:, 1] < 0)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
