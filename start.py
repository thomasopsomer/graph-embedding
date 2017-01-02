# -*- coding: utf-8 -*-
# @Author: ThomasO

import networkx as nx
import numpy as np
from gensim.models import Word2Vec
import random
import logging
import sys
from sklearn import datasets
from sklearn.manifold import TSNE
from collections import defaultdict
from six import iteritems, itervalues
from math import sqrt
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from smart_open import smart_open


# start with karate graph
path = "./karate.edgelist"

# read graph edge list

G = nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph())

# no weight so default weight of 1
for edge in G.edges():
    G[edge[0]][edge[1]]['weight'] = 1

# not directed graph
G = G.to_undirected()


# access nodes, and edges
G.nodes()   # return list of nodes
G.edges()   # return list of edges tuple
G.is_directed()     # shoulb be false :)
# adjacency matrix
M = nx.adjacency_matrix(G)


make_random_walk(G, 2, 5)


def make_random_walk(G, start_node, length, alpha=None):
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


class RandomWalksGenerator(object):
    """ """
    def __init__(self, graph, walk_length, num_walks, p=1, q=1):
        """ """
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

    def __iter__(self, buffsize=1000):
        """ """
        # G.nodes_iter
        for n in range(self.num_walks):
            # nodes_iter
            nodes_iter = G.nodes_iter()
            # for node in node_iter
            for node in itershuffle(nodes_iter, buffsize):
                walk = self._make_RW(self.G, node, self.walk_length)
                yield [str(n) for n in walk]

    def write_walks(self, path):
        with smart_open(path, 'wb') as fout:
            for walk in self:
                fout.write(" ".join(walk))


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


def deep_walk(G, walk_length=10, num_walks=20, dim=2, iter=50):
    RWG = RandomWalksGenerator(G, walk_length=walk_length, num_walks=num_walks)
    skipgram = Word2Vec(sg=1, iter=iter, min_count=0, size=dim, batch_words=100)
    skipgram.build_vocab(RWG)
    skipgram.train(RWG)
    return skipgram

class MyWord2Vec(Word2Vec):
    """
    """
    def build_vocab(self, sentences, keep_raw_vocab=False, trim_rule=None,
                    progress_per=10000, update=False):
        """
        Build vocabulary from a sequence of sentences
            (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.

        """
        self.scan_vocab(sentences, progress_per=progress_per,
                        trim_rule=trim_rule, update=update)
        self.scale_vocab(keep_raw_vocab=keep_raw_vocab,
                         trim_rule=trim_rule, update=update)
        self.finalize_vocab(update=update)

    def scan_vocab(self, sentences, progress_per=10000, trim_rule=None,
                   update=False):
        """Do an initial scan of all words appearing in sentences."""
        logger.info("collecting all nodes and their counts")
        sentence_no = -1
        total_words = 0
        vocab = defaultdict(int)

        for sentence_no, sentence in enumerate(sentences):
            for word in sentence:
                vocab[word] += 1
        total_words += sum(itervalues(vocab))
        logger.info("collected %i word types from a corpus of %i raw words and %i sentences",
                    len(vocab), total_words, sentence_no + 1)
        self.corpus_count = sentence_no + 1
        self.raw_vocab = vocab
        self.total_words = total_words

    def scale_vocab(self, sample=None, dry_run=False,
                    keep_raw_vocab=False, trim_rule=None, update=False):
        """
        Apply vocabulary settings for `min_count` (discarding less-frequent words)
        and `sample` (controlling the downsampling of more-frequent words).

        Calling with `dry_run=True` will only simulate the provided settings and
        report the size of the retained vocabulary, effective corpus length, and
        estimated memory requirements. Results are both printed via logging and
        returned as a dict.

        Delete the raw vocabulary after the scaling is done to free up RAM,
        unless `keep_raw_vocab` is set.

        """
        sample = sample or self.sample

        logger.info("Loading a fresh vocabulary")

        # Discard words less-frequent than min_count
        if not dry_run:
            self.index2word = []
            # make stored settings match these applied settings
            self.sample = sample
            self.vocab = {}

        for word, v in iteritems(self.raw_vocab):
            if not dry_run:
                self.vocab[word] = Vocab(count=v, index=len(self.index2word))
                self.index2word.append(word)

        retain_total = self.total_words

        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        elif sample < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_total
        else:
            # new shorthand: sample >= 1 means downsample all words with
            # higher count than sample
            threshold_count = int(sample * (3 + sqrt(5)) / 2)

        downsample_total, downsample_unique = 0, 0
        for w in self.raw_vocab.iterkeys():
            v = self.raw_vocab[w]
            word_probability = (sqrt(v / threshold_count) + 1) * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                self.vocab[w].sample_int = int(round(word_probability * 2**32))

        if not dry_run and not keep_raw_vocab:
            logger.info("deleting the raw counts dictionary of %i items",
                        len(self.raw_vocab))
            self.raw_vocab = defaultdict(int)

        logger.info("sample=%g downsamples %i most-common words",
                    sample, downsample_unique)
        logger.info("downsampling leaves estimated %i word corpus (%.1f%% of prior %i)",
                    downsample_total, downsample_total * 100.0 / max(retain_total, 1), retain_total)

        # print extra memory estimates
        memory = self.estimate_memory(vocab_size=len(self.vocab))

        return memory

    def finalize_vocab(self, update=False):
        """
        Build tables and model weights based on final vocabulary settings.
        """
        if not self.index2word:
            self.scale_vocab()
        if self.sorted_vocab and not update:
            self.sort_vocab()
        if self.hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree()
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table()
        # set initial input/projection and hidden weights
        if not update:
            self.reset_weights()
        else:
            self.update_weights()


logger = logging.getLogger("gensim")
logger.handlers[0].stream = sys.stdout


RWG = RandomWalksGenerator(G, walk_length=10, num_walks=20)
skipgram = Word2Vec(sg=1, iter=50, min_count=0, size=4, batch_words=100)
skipgram.build_vocab(RWG)
skipgram.train(RWG)
skipgram.init_sims()

tsne = TSNE(n_components=2, random_state=0)


Y1 = tsne.fit_transform(skipgram.syn0)
Y2 = skipgram.syn0


from bokeh.plotting import figure, output_file, show

# output to static HTML file
output_file("line.html")

p = figure(plot_width=400, plot_height=400)
p.circle(Y2[:, 0], Y2[:, 1], color="navy", alpha=0.5)
# show the results
show(p)


#   Two moons

n_samples = 100
X, Y = datasets.make_moons(n_samples=n_samples, noise=.05)

# colors
colors = ["#B3DE69" if y else "#CAB2D6" for y in Y]

# output to static HTML file
output_file("two_moons.html")

p = figure(plot_width=400, plot_height=400)
p.circle(X[:, 0], X[:, 1], color=colors, alpha=0.5)
# show the results
show(p)


from scipy.spatial.distance import pdist, euclidean
from sklearn.metric.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph


# build graph
def RBF(x1, x2, sigma=1):
    """ """
    return np.exp(euclidean(x1, x2) / sigma)


sim_mat = pairwise_distances(X, metric=RBF)


n_neighbors = 10
mode = "distance"   # connectivity or distance
kneighbors_graph(X, n_neighbors=n_neighbors, metric=RBF, mode=mode)










