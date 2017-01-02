# -*- coding: utf-8 -*-
# @Author: ThomasO
from random_walk import RandomWalksGeneratorCSR
from helper import read_csr_matrix
import argparse


parser = argparse.ArgumentParser(description="Compute Random Walks")


parser.add_argument(
    '--input', type=str,
    help=u"path to your edgelist representing the graph")

parser.add_argument(
    '--output', type=str,
    help=u"path a folder to store computed random walk files ")

parser.add_argument(
    '--num-walks', type=int, default=10,
    help=u"Number of walks per node")

parser.add_argument(
    '--walk-length', type=int, default=15,
    help=u"Number of walks per node")

parser.add_argument(
    '--p', type=float, default=1.0,
    help=u"Node2vec parameter p. Default to 1.0")

parser.add_argument(
    '--q', type=float, default=1.0,
    help=u"Node2vec parameter q. Default to 1.0")

parser.add_argument(
    '--make-sym', type=bool, default=False,
    help=u"Flag to duplicate edge symetrically. Make the the undirected")

parser.add_argument(
    '--worker', type=int, default=2,
    help=u"Number of worker to use.")


def main(input, output, make_sym, walk_length, num_walks,
         p, q, worker=4):

    # load karate graph in csr matrix
    csr = read_csr_matrix(input, make_sym=make_sym)

    # init RW generator
    RWG = RandomWalksGeneratorCSR(P=csr, walk_length=walk_length,
                                  num_walks=num_walks, p=p, q=q,
                                  preprocess=True)

    # preprocess transition
    # RWG.preprocess_transition()
    RWG.write_walks(output, worker, writer=1)


if __name__ == '__main__':
    args = parser.parse_args()
    main(**args.__dict__)
