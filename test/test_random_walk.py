# -*- coding: utf-8 -*-
# @Author: ThomasO
from random_walk import RandomWalksGeneratorCSR
from helper import read_csr_matrix


# load karate graph in csr matrix
karate_csr = read_csr_matrix("karate.txt", make_sym=True)

#
p = 2
q = 0.3
walk_length = 15
num_walks = 10

# init RW generator
RWG = RandomWalksGeneratorCSR(P=karate_csr, walk_length=walk_length,
                              num_walks=num_walks, p=2, q=0.5,
                              preprocess=False)

# preprocess transition
RWG.preprocess_transition()

# compute and write Random walks
path = "./RW"
RWG.write_walks(path, worker=4, writer=1, chunck_size=10)

# load random walk generator
RWG = RandomWalksGeneratorCSR(path='./RW')
