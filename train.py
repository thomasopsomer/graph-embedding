# -*- coding: utf-8 -*-
# @Author: ThomasO
from random_walk import RandomWalksGeneratorCSR
from word2vec import Word2Vec
import argparse
import logging


# logging config
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# command line arg parser
parser = argparse.ArgumentParser(
    description="Learn Node Embedding using SkipGram")


parser.add_argument(
    '--input', type=str,
    help=u"path to your edgelist representing the graph")

parser.add_argument(
    '--output', type=str,
    help=u"path a folder to store computed random walk files ")

parser.add_argument(
    '--iter', type=int, default=5,
    help=u"Number iteration over the random walk corpus")

parser.add_argument(
    '--size', type=int, default=128,
    help=u"Size of embedding vectors")

parser.add_argument(
    '--worker', type=int, default=4,
    help=u"Number of worker for the skipgram algorithm")

parser.add_argument(
    '--batch-nodes', type=int, default=10000,
    help=u"Number of node per batch")

parser.add_argument(
    '--negative', type=int, default=5,
    help=u"Value for the negative parameter of word2vec. \
        Number of negative samples")

parser.add_argument(
    '--sample', type=float, default=1e-5,
    help=u"Value for the sample parameter of word2vec. \
        Treshold for downsampling reccurent nodes")

parser.add_argument(
    '--output-format', type=str, default="gensim",
    help=u"Format of the ouput: \
        gensim (gensim.word2vec.Word2Vec picklize model) or \
        txt (standard format in txt file for word2vec)")


def main(input, output, iter=5, size=128, worker=4, batch_nodes=10000,
         negative=5, sample=1e-4, output_format="gensim"):

    # load karate graph in csr matrix
    RWG = RandomWalksGeneratorCSR(path=input)
    # init model
    skipgram = Word2Vec(sg=1, iter=iter, min_count=0, size=size,
                        workers=worker, batch_words=batch_nodes,
                        sample=sample, negative=negative)
    # build vocab
    skipgram.build_vocab(RWG)
    # learn embbeding
    skipgram.train(RWG)
    if output_format == "gensim":
        skipgram.save(output)
    elif output_format == "txt":
        skipgram.save_word2vec_format(output)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**args.__dict__)
