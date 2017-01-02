# -*- coding: utf-8 -*-
# @Author: ThomasO
import random
import numpy as np
import scipy.sparse as sparse


def read_csr_matrix(path, comments="#", delimiter=None, weight=False,
                    make_sym=False):
    """ """
    dtype = [('f0', int), ('f1', int)]
    usecols = (0, 1)
    if weight:
        type.append(('weight', float))
        usecols = (0, 1, 2)
    arr_list = np.loadtxt(path, comments=comments, delimiter=delimiter,
                          dtype=dtype, usecols=usecols, unpack=True)
    if make_sym:
        tmp = arr_list[0]
        arr_list[0] = np.concatenate((arr_list[0], arr_list[1]), axis=0)
        arr_list[1] = np.concatenate((arr_list[1], tmp), axis=0)
        del tmp
    #
    n = max(arr_list[0].max() + 1, arr_list[1].max() + 1)
    if len(usecols) == 2:
        arr_list.append(np.ones([arr_list[0].shape[0]]))
    #
    csr = sparse.csr_matrix((arr_list[2], (arr_list[0], arr_list[1])),
                            shape=(n, n), dtype=float)
    return csr


def itershuffle(iterable, bufsize=1000):
    """
    Shuffle an iterator. This works by holding `bufsize` items back
    and yielding them sometime later. This is NOT 100% random,
    proved or anything.
    """
    iterable = iter(iterable)
    buf = []
    try:
        while True:
            for i in xrange(random.randint(1, bufsize - len(buf))):
                buf.append(iterable.next())
            random.shuffle(buf)
            for i in xrange(random.randint(1, bufsize)):
                if buf:
                    yield buf.pop()
                else:
                    break
    except StopIteration:
        random.shuffle(buf)
        while buf:
            yield buf.pop()
        raise StopIteration
