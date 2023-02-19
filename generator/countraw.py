import numpy as np
import igraph as ig
import os
import json
from collections import Counter
from utils import retrieve_multiple_edges
from utils1 import read_graphs_from_dir, read_patterns_from_dir

INF = float("inf")


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
        '''edge1_labels = set(edges1["label"])
        for el in edges2["label"]:
            if el not in edge1_labels:
                return False'''
        return True

    @classmethod
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

    def check(self, graph, pattern, **kw):
        # valid or not
        if graph.vcount() < pattern.vcount():
            return False
        if graph.ecount() < pattern.ecount():
            return False

        graph_vlabels = Counter(graph.vs["label"])
        pattern_vlabels = Counter(pattern.vs["label"])
        if len(graph_vlabels) < len(pattern_vlabels):
            return False
        for vertex_label, pv_cnt in pattern_vlabels.most_common():
            diff = graph_vlabels[vertex_label] - pv_cnt
            if diff < 0:
                return False

        graph_elabels = Counter(graph.es["label"])
        pattern_elabels = Counter(pattern.es["label"])
        if len(graph_elabels) < len(pattern_elabels):
            return False
        for edge_label, pe_cnt in pattern_elabels.most_common():
            diff = graph_elabels[edge_label] - pe_cnt
            if diff < 0:
                return False
        return True

    def get_subisomorphisms(self, graph, pattern, **kw):
        if not self.check(graph, pattern):
            return list()

        seed_v1 = kw.get("seed_v1", -1)
        seed_v2 = kw.get("seed_v2", -1)
        seed_e1 = kw.get("seed_e1", -1)
        seed_e2 = kw.get("seed_e2", -1)

        vertex_color_vectors = PatternChecker.get_vertex_color_vectors(graph, pattern, seed_v1=seed_v1, seed_v2=seed_v2)
        edge_color_vectors = PatternChecker.get_edge_color_vectors(graph, pattern, seed_e1=seed_e1, seed_e2=seed_e2)

        vertices_in_graph = list()
        if seed_v1 != -1:
            vertices_in_graph.append(seed_v1)
        if seed_e1 != -1:
            vertices_in_graph.extend(graph.es[seed_e1].tuple)
        subisomorphisms = list()  # [(component, mapping), ...]
        for vertex_colors in vertex_color_vectors:
            for edge_colors in edge_color_vectors:
                for subisomorphism in graph.get_subisomorphisms_vf2(pattern,
                                                                    color1=vertex_colors[0], color2=vertex_colors[1],
                                                                    edge_color1=edge_colors[0],
                                                                    edge_color2=edge_colors[1],
                                                                    node_compat_fn=PatternChecker.node_compat_fn,
                                                                    edge_compat_fn=PatternChecker.edge_compat_fn):
                    if len(vertices_in_graph) == 0 or all([v in subisomorphism for v in vertices_in_graph]):
                        subisomorphisms.append(subisomorphism)
        return subisomorphisms

    def count_subisomorphisms(self, graph, pattern, **kw):
        if not self.check(graph, pattern):
            return 0

        seed_v1 = kw.get("seed_v1", -1)
        seed_v2 = kw.get("seed_v2", -1)
        seed_e1 = kw.get("seed_e1", -1)
        seed_e2 = kw.get("seed_e2", -1)

        vertex_color_vectors = PatternChecker.get_vertex_color_vectors(graph, pattern, seed_v1=seed_v1, seed_v2=seed_v2)
        edge_color_vectors = PatternChecker.get_edge_color_vectors(graph, pattern, seed_e1=seed_e1, seed_e2=seed_e2)

        vertices_in_graph = list()
        if seed_v1 != -1:
            vertices_in_graph.append(seed_v1)
        if seed_e1 != -1:
            vertices_in_graph.extend(graph.es[seed_e1].tuple)
        if len(vertices_in_graph) == 0:
            counts = 0
            for vertex_colors in vertex_color_vectors:
                for edge_colors in edge_color_vectors:
                    counts += graph.count_subisomorphisms_vf2(pattern,
                                                              color1=vertex_colors[0], color2=vertex_colors[1],
                                                              edge_color1=edge_colors[0], edge_color2=edge_colors[1],
                                                              node_compat_fn=PatternChecker.node_compat_fn,
                                                              edge_compat_fn=PatternChecker.edge_compat_fn)
            return counts
        else:
            counts = 0
            for vertex_colors in vertex_color_vectors:
                for edge_colors in edge_color_vectors:
                    for subisomorphism in graph.get_subisomorphisms_vf2(pattern,
                                                                        color1=vertex_colors[0],
                                                                        color2=vertex_colors[1],
                                                                        edge_color1=edge_colors[0],
                                                                        edge_color2=edge_colors[1],
                                                                        node_compat_fn=PatternChecker.node_compat_fn,
                                                                        edge_compat_fn=PatternChecker.edge_compat_fn):
                        if all([v in subisomorphism for v in vertices_in_graph]):
                            counts += 1
            return counts


if __name__ == "__main__":
    num_workers=16
    #pattern_path="../data/NCI1/useful_patterns"
    #metadata_path="../data/NCI1/useful_metadata"
    '''count=0
    ground=0
    pair_n=0
    for filename in os.listdir(pattern_path):
        pattern = ig.read(os.path.join(pattern_path, filename))
        pattern_name,ext=os.path.splitext(filename)
        save_metadata_dir=os.path.join(os.path.join(metadata_path,pattern_name))
        if os.path.exists(save_metadata_dir)==False:
            os.mkdir(save_metadata_dir)
        for filenameg in os.listdir(graph_path):
            graph=ig.read(os.path.join(graph_path,filenameg))
            graph_name, ext = os.path.splitext(filenameg)
            ground_truth = graph.count_subisomorphisms_vf2(pattern,
                                                           node_compat_fn=PatternChecker.node_compat_fn,
                                                           edge_compat_fn=PatternChecker.edge_compat_fn)

            pc = PatternChecker()
            pair_n+=1
            if ground_truth!=0:
                count+=1
                ground+=ground_truth
            metadata = {"counts": ground_truth, "subisomorphisms": pc.get_subisomorphisms(graph, pattern)}
            #metadata=json.dump(metadata)
            with open(os.path.join(save_metadata_dir, graph_name+ ".meta"), "w") as f:
                json.dump(metadata, f)
print(count)
print(pair_n)
print(ground)
print(ground/pair_n)'''
    graph_path="../dataset/ppa/raw"

    max_n=0
    max_e=0
    max_nl=0
    max_el=0
    for filenameg in os.listdir(graph_path):
        graph = ig.read(os.path.join(graph_path, filenameg))
        n=graph.vcount()
        e=graph.ecount()
        nl = np.max(graph.vs["label"])
        el = np.max(graph.es["label"])
        if max_n<n :
            max_n=n
        if max_e<e:
            max_e=e
        if max_nl<nl:
            max_nl=nl
        if max_el<el:
            max_el=el
    print('max_n: ', max_n)
    print('max_e: ', max_e)
    print('max_nl: ', max_nl)
    print('max_el: ', max_el)


