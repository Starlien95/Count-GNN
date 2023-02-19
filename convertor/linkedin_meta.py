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
    node2graph=list()
    graphnum=0
    with open(os.path.join(dblp_data_path, "linkedin.metagraph.db"), "r") as f:
        for line, content in enumerate(f):
            if(content[0]=='#'):
                graphnum+=1
            elif(content[0]=='T'):
                content=content.strip('\n')
                content=content.strip('T')
                content=content.split('\t')
                content.remove(content[0])
                node=list()
                nodelabel=list()
                i=0
                for n in content:
                    nodelabel.append(int(n))
                    node.append(i)
                    i+=1
                nodes.append(node)
                nodelabels.append(nodelabel)
            elif(content[0]=='E'):
                content=content.strip('\n')
                content=content.strip('E')
                content=content.split('\t')
                content.remove(content[0])
                edge=list()
                u=v=0
                i=0
                for n in content:
                    if(i%2==0):
                        u=int(n)
                    else:
                        v=int(n)
                        edge.append((u,v))
                    i=i+1
                edges.append(edge)

    os.makedirs(os.path.join(dblp_data_path, "pattern"), exist_ok=True)
    max_vcount=max_vlabels=max_ecount=0
    for g_id in tqdm(range(graphnum)):
        graph = ig.Graph(directed=True)
        vcount = len(nodes[g_id])
        vlabels = nodelabels[g_id]
        #elabels = edgelabels[g_id]

        graph.add_vertices(vcount)
        graph.add_edges(edges[g_id])
        graph.vs["label"] = vlabels
        elables=list()
        for i in range(len(edges[g_id])):
            elables.append(0)
        graph.es["label"]=elables
        graph.es["key"] = [0] * len(edges[g_id])

        if(vcount>max_vcount):
            max_vcount=vcount
        if(len(edges[g_id])>max_ecount):
            max_ecount=len(edges[g_id])
        if(max(vlabels)>max_vlabels):
            max_vlabels=max(vlabels)
        graph_id = "P_N%d_E%d_NL%d_%d" % (
            vcount, len(edges[g_id]), max(vlabels) + 1, g_id)
        filename = os.path.join(dblp_data_path, "pattern", graph_id)
        graph.write(filename + ".gml")
    print(max_vcount,max_vlabels+1,max_ecount)