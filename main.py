# -*- coding: utf-8 -*-
# @Author: ThomasO
from graph_embbeding import Node2Vec
import networkx as nx
import begin


@begin.start
@begin.convert(weighted=bool, to_undirected=bool)
def main(input_path=None, output_path=None, weighted=False, to_undirected=True,
         p=1, q=1, walk_length=10, num_walk=10, epoch=10, size=128,
         batch_words=1000):
    """ """
    c = locals()
    # load graph
    G = nx.read_edgelist(input_path, nodetype=int,
                         create_using=nx.DiGraph())
    if not weighted:
        # no weight so default weight of 1
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if to_undirected:
        # not directed graph
        G = G.to_undirected()

    # model
    graph_emb = Node2Vec(G, walk_length=walk_length, num_walk=num_walk,
                         p=p, q=q, epoch=epoch, size=size,
                         batch_words=batch_words)

    # save word2vec format
    graph_emb.sg.save_word2vec_format(output_path)
    with open(output_path + ".config", "w") as conf:
        conf.write(str(c))






