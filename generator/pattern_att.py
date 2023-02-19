import numpy as np
import igraph as ig
import os
import json
from collections import Counter
from utils import retrieve_multiple_edges
from utils1 import read_graphs_from_dir, read_patterns_from_dir

pattern_path = "../data/NCI1/newpatterns"
metadata_path = "../data/NCI1/newmetadata"
max_ecount=0
max_vlabel=0
for filename in os.listdir(pattern_path):
    pattern = ig.read(os.path.join(pattern_path, filename))
    if max_ecount<pattern.ecount():
        max_ecount=pattern.ecount()

print(max_ecount)