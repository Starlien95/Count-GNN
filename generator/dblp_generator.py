import igraph as ig
import numpy as np
import os
import time
import json
from collections import Counter
from utils import retrieve_multiple_edges
from multiprocessing import Pool



def issubset(s1, s2):
    if len(s1) < len(s2):
        return False
    return len(s2 - s1) == 0


class PatternChecker(object):
    dummy_label = -1

    def __init__(self):
        pass

    @classmethod
    def node_compat_fn(cls, g1, g2, v1, v2):
        if "label" not in g2.vertex_attributes():
            return True
        vertex2 = g2.vs[v2]
        if vertex2["label"] == cls.dummy_label:
            return True
        vertex1 = g1.vs[v1]
        if vertex1["label"] != vertex2["label"]:
            return False

        # for loop
        edges1 = retrieve_multiple_edges(g1, v1, v1)
        edges2 = retrieve_multiple_edges(g2, v2, v2)
        if len(edges1) < len(edges2):
            return False
        return issubset(set(edges1["label"]), set(edges2["label"]))

    @classmethod
    def edge_compat_fn(cls, g1, g2, e1, e2):
        if "label" not in g2.edge_attributes():
            return True
        edge2 = g2.es[e2]
        if edge2["label"] == cls.dummy_label:
            return True

        edge1 = g1.es[e1]
        # for multiedges
        edges1 = retrieve_multiple_edges(g1, edge1.source, edge1.target)
        edges2 = retrieve_multiple_edges(g2, edge2.source, edge2.target)
        if len(edges1) < len(edges2):
            return False
        return issubset(set(edges1["label"]), set(edges2["label"]))

    @classmethod
    def get_vertex_color_vectors(cls, g1, g2, seed_v1=None, seed_v2=None):
        dm = cls.dummy_label
        N1 = g1.vcount()
        N2 = g2.vcount()
        color_vectors = list()

        color1 = g1.vs[seed_v1]["label"] if seed_v1 is not None else dm
        color2 = g2.vs[seed_v2]["label"] if seed_v2 is not None else dm

        if color1 == dm and color2 == dm:
            color_vectors.append((None, None))
        elif color1 != dm and color2 != dm:
            if color1 == color2:
                color1 = np.zeros((N1,), dtype=np.int64)
                color2 = np.zeros((N2,), dtype=np.int64)
                color_vectors.append((color1, color2))
        elif color1 != dm:
            seed_label = color1
            color1 = np.zeros((N1,), dtype=np.int64)
            color1[seed_v1] = 1
            for seed_v2, vertex in enumerate(g2.vs):
                if vertex["label"] == seed_label:
                    color2 = np.zeros((N2,), dtype=np.int64)
                    color2[seed_v2] = 1
                    color_vectors.append((color1, color2))
        else:  # color2 != dm
            seed_label = color2
            color2 = np.zeros((N2,), dtype=np.int64)
            color2[seed_v2] = 1
            for seed_v1, vertex in enumerate(g1.vs):
                if vertex["label"] == seed_label:
                    color1 = np.zeros((N1,), dtype=np.int64)
                    color1[seed_v1] = 1
                    color_vectors.append((color1, color2))
        return color_vectors

    @classmethod
    def get_edge_color_vectors(cls, g1, g2, seed_e1=None, seed_e2=None):
        dm = cls.dummy_label
        E1 = len(g1.es)
        E2 = len(g2.es)
        edge_color_vectors = list()

        color1 = g1.es[seed_e1]["label"] if seed_e1 is not None else dm
        color2 = g2.es[seed_e2]["label"] if seed_e2 is not None else dm

        if color1 == dm and color2 == dm:
            edge_color_vectors.append((None, None))
        elif color1 != dm and color2 != dm:
            if color1 == color2 and g1.es[seed_e1].is_loop() == g2.es[seed_e2].is_loop():
                edge_color_vectors.append((color1, color2))
        elif color1 != dm:
            seed_label = color1
            is_loop = g1.es[seed_e1].is_loop()
            color1 = np.zeros((E1,), dtype=np.int64)
            color1[seed_e1] = 1
            for seed_e2, edge in enumerate(g2.es):
                if edge["label"] == seed_label and is_loop == edge.is_loop():
                    color2 = np.zeros((E2,), dtype=np.int64)
                    color2[seed_e2] = 1
                    edge_color_vectors.append((color1, color2))
        else:  # color2 != dm:
            seed_label = color2
            is_loop = g2.es[seed_e2].is_loop()
            color2 = [0] * E2
            color2[seed_e2] = 1
            for seed_e1, edge in enumerate(g1.es):
                if edge["label"] == seed_label and is_loop == edge.is_loop():
                    color1 = [0] * E1
                    color1[seed_e1] = 1
                    edge_color_vectors.append((color1, color2))

        return edge_color_vectors

    def check(self, graph, pattern, **kw):
        # valid or not
        if graph.vcount() < pattern.vcount():
            return False
        if graph.ecount() < pattern.ecount():
            return False

        if "label" in pattern.vertex_attributes():
            graph_vlabels = Counter(graph.vs["label"])
            pattern_vlabels = Counter(pattern.vs["label"])
            if len(graph_vlabels) < len(pattern_vlabels):
                return False
            for vertex_label, pv_cnt in pattern_vlabels.most_common():
                diff = graph_vlabels[vertex_label] - pv_cnt
                if diff < 0:
                    return False

        if "label" in pattern.edge_attributes():
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
                for subisomorphism in graph.get_subisomorphisms_vf2(
                    pattern,
                    color1=vertex_colors[0],
                    color2=vertex_colors[1],
                    edge_color1=edge_colors[0],
                    edge_color2=edge_colors[1],
                    node_compat_fn=PatternChecker.node_compat_fn,
                    edge_compat_fn=PatternChecker.edge_compat_fn
                ):
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
                    counts += graph.count_subisomorphisms_vf2(
                        pattern,
                        color1=vertex_colors[0],
                        color2=vertex_colors[1],
                        edge_color1=edge_colors[0],
                        edge_color2=edge_colors[1],
                        node_compat_fn=PatternChecker.node_compat_fn,
                        edge_compat_fn=PatternChecker.edge_compat_fn
                    )
            return counts
        else:
            counts = 0
            for vertex_colors in vertex_color_vectors:
                for edge_colors in edge_color_vectors:
                    for subisomorphism in graph.get_subisomorphisms_vf2(
                        pattern,
                        color1=vertex_colors[0],
                        color2=vertex_colors[1],
                        edge_color1=edge_colors[0],
                        edge_color2=edge_colors[1],
                        node_compat_fn=PatternChecker.node_compat_fn,
                        edge_compat_fn=PatternChecker.edge_compat_fn
                    ):
                        if all([v in subisomorphism for v in vertices_in_graph]):
                            counts += 1
            return counts

def generate(filename, graph_path,metadata_path):
    pattern = ig.read(os.path.join(pattern_path, filename))
    pattern_name,ext=os.path.splitext(filename)
    save_metadata_dir=os.path.join(os.path.join(metadata_path,pattern_name))
    if os.path.exists(save_metadata_dir)==False:
        os.mkdir(save_metadata_dir)
    for filenameg in os.listdir(graph_path):
        graph=ig.read(os.path.join(graph_path,filenameg))
        graph_name, ext = os.path.splitext(filenameg)
        pc = PatternChecker()
        #ts=time.time()
        ground_truth = graph.count_subisomorphisms_vf2(pattern,
                                                       node_compat_fn=PatternChecker.node_compat_fn,
                                                       edge_compat_fn=PatternChecker.edge_compat_fn)
        #te=time.time()
        #get_iso = graph.get_subisomorphisms_vf2(pattern, node_compat_fn=PatternChecker.node_compat_fn,
        #                                        edge_compat_fn=PatternChecker.edge_compat_fn)
        get_iso=list()
        # metadata = {"counts": ground_truth, "subisomorphisms": pc.get_subisomorphisms(graph, pattern)}
        metadata = {"counts": ground_truth, "subisomorphisms": get_iso}
        #metadata=json.dump(metadata)
        with open(os.path.join(save_metadata_dir, graph_name+ ".meta"), "w") as f:
            json.dump(metadata, f)
        print('done')
        #return te-ts
if __name__ == "__main__":
    num_workers=16
    pattern_path="../lgraph/dblp/pattern"
    graph_path="../lgraph/dblp/raw"
    metadata_path="../lgraph/dblp/metadata"
    pool=Pool(num_workers)
    ts=time.time()
    for filename in os.listdir(pattern_path):
        #t+=pool.apply_async(generate,args=(filename,graph_path,metadata_path)).get()
        pool.apply_async(generate, args=(filename, graph_path, metadata_path))
    te=time.time()
    pool.close()
    pool.join()
    print(te-ts)
    '''for filename in os.listdir(pattern_path):
        pattern = ig.read(os.path.join(pattern_path, filename))
        pattern_name,ext=os.path.splitext(filename)
        save_metadata_dir=os.path.join(os.path.join(metadata_path,pattern_name))
        if os.path.exists(save_metadata_dir)==False:
            os.mkdir(save_metadata_dir)
        for filenameg in os.listdir(graph_path):
            graph=ig.read(os.path.join(graph_path,filenameg))
            graph_name, ext = os.path.splitext(filenameg)
            pc = PatternChecker()
            times=time.time()
            ground_truth = graph.count_subisomorphisms_vf2(pattern,
                                                           node_compat_fn=PatternChecker.node_compat_fn,
                                                           edge_compat_fn=PatternChecker.edge_compat_fn)
            #get_iso = graph.get_subisomorphisms_vf2(pattern, node_compat_fn=PatternChecker.node_compat_fn,
            #                                        edge_compat_fn=PatternChecker.edge_compat_fn)
            get_iso=list()
            # metadata = {"counts": ground_truth, "subisomorphisms": pc.get_subisomorphisms(graph, pattern)}
            metadata = {"counts": ground_truth, "subisomorphisms": get_iso}
            #metadata=json.dump(metadata)
            with open(os.path.join(save_metadata_dir, graph_name+ ".meta"), "w") as f:
                json.dump(metadata, f)'''



    '''graph = ig.read("../data/lgraph/dblp/raw/G_N172136_E1937644_NL5_0.gml")
    pattern  = ig.read("../data/lgraph/dblp/pattern/P_N4_E8_NL2_0.gml")
    #pattern  = ig.read("../data/lgraph/dblp/pattern/test.gml")
    ground_truth = graph.count_subisomorphisms_vf2(
        pattern, node_compat_fn=PatternChecker.node_compat_fn, edge_compat_fn=PatternChecker.edge_compat_fn)
    #get_iso=graph.get_subisomorphisms_vf2(pattern, node_compat_fn=PatternChecker.node_compat_fn,
    #                                      edge_compat_fn=PatternChecker.edge_compat_fn)
    get_iso=list()
    #metadata = {"counts": ground_truth, "subisomorphisms": pc.get_subisomorphisms(graph, pattern)}
    metadata = {"counts": ground_truth, "subisomorphisms": get_iso}
    print(metadata)'''


