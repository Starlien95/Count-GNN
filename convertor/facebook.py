import igraph as ig
import os
import sys
import torch
from tqdm import tqdm

if __name__ == "__main__":
    assert len(sys.argv) == 2
    dblp_data_path = sys.argv[1]

    nodes = list()
    edges=list()
    nodelabels = list()
    with open(os.path.join(dblp_data_path, "facebook.lg"), "r") as f:
        for line, content in enumerate(f):
            if(content[0]=='v'):
                content=content.strip('\n')
                content=content.split(' ')
                nodes.append(int(content[1]))
                nodelabels.append(int(content[2]))
            elif(content[0]=='e'):
                content=content.strip('\n')
                content=content.split(' ')
                edges.append((int(content[1]),int(content[2])))
                edges.append((int(content[2]),int(content[1])))
                #edges[0].append(int(content[1]))
                #edges[1].append(int(content[2]))

    os.makedirs(os.path.join(dblp_data_path, "raw"), exist_ok=True)
    max_vcount=max_vlabels=max_ecount=0
    graph = ig.Graph(directed=True)
    vcount = len(nodes)
    vlabels = nodelabels
    # elabels = edgelabels[g_id]

    graph.add_vertices(vcount)
    graph.add_edges(edges)
    graph.vs["label"] = vlabels
    elables = list()
    for i in range(len(edges)):
        elables.append(0)
    graph.es["label"] = elables
    graph.es["key"] = [0] * len(edges)

    graph_id = "G_N%d_E%d_NL%d_EL%d_%d" % (
        vcount, len(edges), max(vlabels) + 1, 1, 0)
    if (vcount > max_vcount):
        max_vcount = vcount
    if (len(edges) > max_ecount):
        max_ecount = len(edges)
    if (max(vlabels) > max_vlabels):
        max_vlabels = max(vlabels)
    graph_id = "G_N%d_E%d_NL%d_%d" % (
        vcount, len(edges), max(vlabels) + 1, 0)
    filename = os.path.join(dblp_data_path, "raw", graph_id)
    graph.write(filename + ".gml")

    print(max_vcount,max_vlabels+1,max_ecount)