import numpy as np
import igraph as ig
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import time


def retrieve_multiple_edges(graph, source=-1, target=-1):
    if source != -1:
        e = graph.incident(source, mode=ig.OUT)
        if target != -1:
            e = set(e).intersection(graph.incident(target, mode=ig.IN))
        return ig.EdgeSeq(graph, e)
    else:
        if target != -1:
            e = graph.incident(target, mode=ig.IN)
        else:
            e = list()
        return ig.EdgeSeq(graph, e)


class PatternChecker(object):
    def __init__(self):
        pass

    @classmethod
    def node_compat_fn(cls, g1, g2, v1, v2):
        vertex1 = g1.vs[v1]
        vertex2 = g2.vs[v2]
        return vertex1["label"] == vertex2["label"]

    @classmethod
    def edge_compat_fn(cls, g1, g2, e1, e2):
        edge1 = g1.es[e1]
        edge2 = g2.es[e2]
        if edge1.is_loop() != edge2.is_loop():
            return False
        # for multiedges
        edges1 = retrieve_multiple_edges(g1, edge1.source, edge1.target)
        edges2 = retrieve_multiple_edges(g2, edge2.source, edge2.target)
        if len(edges1) < len(edges2):
            return False
        edge1_labels = set(edges1["label"])
        for el in edges2["label"]:
            if el not in edge1_labels:
                return False
        return True


def get_vertex_color_vectors(cls, g1, g2, seed_v1=-1, seed_v2=-1):
    N1 = g1.vcount()
    N2 = g2.vcount()
    color_vectors = list()
    if seed_v1 == -1 and seed_v2 == -1:
        color_vectors.append((None, None))
    elif seed_v1 == -1 and seed_v2 != -1:
        vertex = g1.vs[seed_v1]
        seed_label = vertex["label"]
        for seed_v1, vertex in enumerate(g1.vs):
            if vertex["label"] == seed_label:
                color1 = [0] * N1
                color1[seed_v1] = 1
                color2 = [0] * N2
                color2[seed_v2] = 1
                color_vectors.append((color1, color2))
    elif seed_v1 != -1 and seed_v2 == -1:
        seed_label = g1.vs[seed_v1]["label"]
        for seed_v2, vertex in enumerate(g2.vs):
            if vertex["label"] == seed_label:
                color1 = [0] * N1
                color1[seed_v1] = 1
                color2 = [0] * N2
                color2[seed_v2] = 1
                color_vectors.append((color1, color2))
    else:  # seed_v1 != -1 and seed_v2 != -1:
        if g1.vs[seed_v1]["label"] == g2.vs[seed_v2]["label"]:
            color1 = [0] * N1
            color1[seed_v1] = 1
            color2 = [0] * N2
            color2[seed_v2] = 1
            color_vectors.append((color1, color2))
    return color_vectors


@classmethod
def get_edge_color_vectors(cls, g1, g2, seed_e1=-1, seed_e2=-1):
    E1 = len(g1.es)
    E2 = len(g2.es)
    edge_color_vectors = list()
    if seed_e1 == -1 and seed_e2 == -1:
        edge_color_vectors.append((None, None))
    elif seed_e1 == -1 and seed_e2 != -1:
        edge = g2.es[seed_e2]
        color2 = [0] * E2
        color2[seed_e2] = 1
        seed_label = edge["label"]
        is_loop = edge.is_loop()
        for seed_e1, edge in enumerate(g1.es):
            if edge["label"] == seed_label and is_loop == edge.is_loop():
                color1 = [0] * E1
                color1[seed_e1] = 1
                edge_color_vectors.append((color1, color2))
    elif seed_e1 != -1 and seed_e2 == -1:
        edge = g1.es[seed_e1]
        color1 = [0] * E1
        color1[seed_e1] = 1
        seed_label = edge["label"]
        is_loop = edge.is_loop()
        for seed_e2, edge in enumerate(g2.es):
            if edge["label"] == seed_label and is_loop == edge.is_loop():
                color2 = [0] * E2
                color2[seed_e2] = 1
                edge_color_vectors.append((color1, color2))
    else:  # seed_e1 != -1 and seed_e2 != -1:
        edge1 = g1.es[seed_e1]
        edge2 = g2.es[seed_e2]
        color1 = [0] * E1
        color1[seed_e1] = 1
        color2 = [0] * E2
        color2[seed_e2] = 1
        if edge1["label"] == edge2["label"] and edge1.is_loop() == edge2.is_loop():
            edge_color_vectors.append((color1, color2))
    return edge_color_vectors

def dgl2igraph(pattern):
    g=ig.Graph()
    g.add_vertices(pattern.number_of_nodes())
    edges=list()
    for i in range(pattern.number_of_edges()):
        x=pattern.adjacency_matrix()._indices()[0][i].item()
        y=pattern.adjacency_matrix()._indices()[1][i].item()
        edges.append((x,y))
    g.add_edges(edges)
    #print(pattern.ndata["label"])
    '''for i in pattern.ndata["label"]:
        g.vs["label"]=i
    print(g.vs["label"])
    for i in pattern.edata["label"][0]:
        g.es["label"]=i'''
    g.vs["label"]=pattern.ndata["label"].numpy().tolist()
    g.es["label"]=pattern.edata["label"].numpy().tolist()
    #print(g.vs["label"])

    return g

def vf2(pattern,graph):
    ig_pattern=dgl2igraph(pattern)
    ig_graph=dgl2igraph(graph)
    s=time.time()
    result = ig_graph.count_subisomorphisms_vf2(ig_pattern,
                                    node_compat_fn=PatternChecker.node_compat_fn,
                                    edge_compat_fn=PatternChecker.edge_compat_fn)
    e=time.time()
    t=e-s
    return t
