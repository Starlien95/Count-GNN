'''calculate edge adj matric for each pattern-graph offline, so that we can speed out model:GATNeigh_Agg'''
import os
import tqdm
import numpy
from utils import GetEdgeAdj


#######################################
#################debug dataset#########
#######################################
for batch in tqdm.trange(112):
    for j in range(8):
        p_eadj=GetEdgeAdj(pattern["train"][batch*8+j]).numpy()
        g_eadj=GetEdgeAdj(graph["train"][batch*8+j]).numpy()
        numpy.savez(os.path.join("./data/debug/edgeadj/train", "%s_edgeadj" % (pattern["train"][batch*8+j].id)),p_eadj,g_eadj)

for batch in tqdm.trange(112):
    p_eadj=GetEdgeAdj(pattern["test"][batch]).numpy()
    g_eadj=GetEdgeAdj(graph["test"][batch]).numpy()
    numpy.savez(os.path.join("./data/debug/edgeadj/test", "%s_edgeadj" % (pattern["test"][batch].id)),p_eadj,g_eadj)


for batch in tqdm.trange(112):
    p_eadj=GetEdgeAdj(pattern["dev"][batch]).numpy()
    g_eadj=GetEdgeAdj(graph["dev"][batch]).numpy()
    numpy.savez(os.path.join("./data/debug/edgeadj/dev", "%s_edgeadj" % (pattern["dev"][batch].id)),p_eadj,g_eadj)

