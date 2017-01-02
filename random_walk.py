# -*- coding: utf-8 -*-
# @Author: ThomasO
from functools import partial
from smart_open import smart_open
import numpy as np
import random
from helper import itershuffle
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import multiprocessing as mp
import os
import sys


class RandomWalksGeneratorNX(object):
    """ """

    def __init__(self, graph, walk_length, num_walks,
                 p=1, q=1):
        """ """
        # handle nx graph
        self.G = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        if p == 1 and q == 1:
            self._make_RW = make_random_walk
        else:
            self._make_RW = partial(make_random_walk_biased,
                                    p=self.p, q=self.q)

    def __iter__(self, shuffle=True, buffsize=1000):
        """ """
        # G.nodes_iter
        for n in range(self.num_walks):
            # nodes_iter
            nodes_iter = self.G.nodes_iter()
            if shuffle:
                it = itershuffle(nodes_iter, buffsize)
            else:
                it = nodes_iter
            # for node in node_iter
            for node in it:
                walk = self._make_RW(self.G, node, self.walk_length)
                yield [str(n) for n in walk]

    def write_walks(self, path):
        with smart_open(path, 'wb') as fout:
            for walk in self:
                fout.write(" ".join(walk))


class RandomWalksGeneratorCSR(object):
    """ """

    def __init__(self, P=None, path=None, walk_length=15, num_walks=10,
                 p=1, q=1, preprocess=False):
        """ """
        if path is not None:
            self.path = path
        if P is not None:
            self.P = P
            self.walk_length = walk_length
            self.num_walks = num_walks
            self.p = p
            self.q = q
            self.n = P.shape[0]
            # preprocess transition for node2vec if needed
            if preprocess:
                self.P = preprocess_transition(P, p=p, q=q)
        if P is None and path is None:
            raise ValueError(
                "Need a csr matrix P or a path to file that containes RWs")
        #
        self._make_RW = make_random_walk_csr

    def preprocess_transition(self):
        """ """
        self.P = preprocess_transition(self.P, p=self.p, q=self.q)

    def __iter__(self, buffsize=1000, shuffle=True):
        """ """
        # if walk already computed and stored locally
        if hasattr(self, "path"):
            for f in filter(lambda x: not x.startswith("."),
                            os.listdir(self.path)):
                fpath = os.path.join(self.path, f)
                with open(fpath) as f:
                    for line in f:
                        walk = line.split()
                        yield walk
        # otherwise compute it on the fly
        # slower but ok for small graph
        else:
            # G.nodes_iter
            for n in range(self.num_walks):
                # nodes_iter
                nodes_iter = xrange(self.P.shape[0])
                if shuffle:
                    it = itershuffle(nodes_iter, buffsize)
                else:
                    it = nodes_iter
                # for node in node_iter
                for node in it:
                    walk = self._make_RW(self.P, node, self.walk_length)
                    yield [str(n) for n in walk]

    def write_walks(self, path, chunck_size=10000, worker=4, writer=1):
        """ """
        # check if path exist
        if not os.path.exists(path):
            os.mkdir(path)

        # queue object
        input_q = mp.JoinableQueue()
        output_q = mp.JoinableQueue()
        # unpack
        P = self.P
        walk_length = self.walk_length
        # instantiate workers
        rwg_workers = [RWGWorker(input_q, output_q, walk_length,
                                 P.data, P.indices, P.indptr, P.shape)
                       for i in range(worker)]
        # instantiate writers
        writers = [Writer(task_queue=output_q, path_root=path)
                   for i in range(writer)]
        workers = rwg_workers + writers

        # start workers
        for w in workers:
            w.start()

        # give work to do
        n = self.n
        n_chunck = n // chunck_size
        for k in xrange(self.num_walks):
            s = 0
            for k in xrange(n_chunck):
                input_q.put(range(s, s + chunck_size))
                s = s + chunck_size
            input_q.put(range(s, n))

        # Add a poison pill for each consumer
        for i in xrange(len(workers)):
            input_q.put(None)

        # Wait for all of the tasks to finish
        # output_q.join()
        # input_q.join()
        for w in workers:
            w.join()
        # set the rootpath as attribute
        self.path = path


def make_random_walk_nx(G, start_node, length):
    """ """
    # first node of the RW
    walk = [start_node]
    #
    while len(walk) < length:
        current_node = walk[-1]
        neighbors = G.neighbors(current_node)
        d = G.degree(current_node, weight="weight")
        p = [1.0 * G[current_node][neighbor]["weight"] / d
             for neighbor in neighbors]
        walk.append(int(np.random.choice(neighbors, 1, p=p)[0]))
    return walk


def make_random_walk_csr(P, start_node, length):
    """
    Assusme that P is the probability transition matrix
    """
    # first node of the RW
    walk = [start_node]
    while len(walk) < length:
        current_node = walk[-1]
        nbrs = P.getrow(current_node)
        nbrs_idx = nbrs.indices
        if len(nbrs_idx) == 0:
            break
        p = nbrs.data
        walk.append(int(np.random.choice(nbrs_idx, p=p)))
    return walk


def alpha(p, q, t, x, P):
    if t == x:
        return 1.0 / p
    elif P[t, x] > 0:
        return 1.0
    else:
        return 1.0 / q


def preprocess_transition(M, p=1, q=1):
    """ """
    # deep walk case
    if p == 1 and q == 1:
        P = normalize(M, norm='l1', axis=1)
    # node2vec case
    else:
        P = M.copy()
        for src in xrange(M.shape[0]):
            src_row = M.getrow(src)
            src_nbr = src_row.indices
            for dst in src_nbr:
                dst_row = M.getrow(dst)
                dst_nbr = dst_row.indices
                dst_prob = dst_row.data
                dst_ptr = dst_row.indptr
                # compute coef alpha of transition
                alphas = np.array([alpha(p, q, src, x, P) for x in dst_nbr])
                dst_prob = alphas * dst_prob
                # update transition matrix P
                P[dst] = csr_matrix((dst_prob, dst_nbr, dst_ptr),
                                    shape=(1, P.shape[1]))
        # normalize probability
        P = normalize(P, norm='l1', axis=1)
    return P


def make_random_walk_biased(G, start_node, length, p, q):
    """ """
    # first node of the RW
    walk = [start_node]

    while len(walk) < length:
        current_node = walk[-1]
        neighbors = G.neighbors(current_node)
        if len(walk) == 1:
            d = G.degree(current_node, weight="weight")
            prob = [1.0 * G[current_node][neighbor]["weight"] / d
                    for neighbor in neighbors]
        else:
            prev = walk[-2]
            prob = []
            for nbr in neighbors:
                if nbr == prev:
                    pr = 1.0 * G[current_node][nbr]["weight"] / p
                elif G.has_edge(prev, nbr):
                    pr = 1.0 * G[current_node][nbr]["weight"]
                else:
                    pr = 1.0 * G[current_node][nbr]["weight"] / q
                prob.append(pr)
            # normalize
            prob = np.array(prob) / sum(prob)

        # sample a next node according to prob
        walk.append(int(np.random.choice(neighbors, 1, p=prob)[0]))
    return walk


class RWGWorker(mp.Process):
    """ Worker that generate RW """
    def __init__(self, input_q, output_q, walk_length, data, indices,
                 indptr, shape, shuffle=False):
        """ """
        super(RWGWorker, self).__init__()

        # instantiate the CSR matrix
        self.P = csr_matrix((data, indices, indptr), shape=shape, copy=False)
        self.input_q = input_q
        self.output_q = output_q
        self.walk_length = walk_length
        self.shuffle = shuffle

    def run(self):
        """ """
        # nodes_iter
        # node = self.input_q.get()
        # if node is not None:
        try:
            for list_of_nodes in iter(self.input_q.get, None):
                # it = itershuffle(xrange(self.P.shape[0]), bufsize=10000)
                if list_of_nodes is not None:
                    random.shuffle(list_of_nodes)
                    for node in list_of_nodes:
                        walk = make_random_walk_csr(self.P, node, self.walk_length)
                        if len(walk) > 2:
                            walk = [str(node) for node in walk]
                            # send result in output queue
                            self.output_q.put(walk)
                self.input_q.task_done()
            #
            self.output_q.put(None)
        except KeyboardInterrupt:
            sys.exit(0)


class Writer(mp.Process):
    """ """
    c = 0

    def __init__(self, task_queue, path_root):
        """ """
        super(Writer, self).__init__()
        self.task_queue = task_queue
        self.path_root = path_root
        self.id = self._count()
        self.path = os.path.join(path_root, "part-%s" % self.c)

    def run(self):
        """ """
        try:
            proc_name = self.name
            with open(self.path, "w") as fout:
                for walk in iter(self.task_queue.get, None):
                    fout.write(" ".join(walk) + "\n")
                    self.task_queue.task_done()
            return
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception, e:
            print e.message, e.args

    @classmethod
    def _count(cls):
        Writer.c += 1
        return Writer.c
