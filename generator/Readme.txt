pattern_generator.py：生成文件夹pattern保存生成的模式图

metag_generator.py:这里生成的图就是最终的graph（graph_generator中有完成subgraph融合的过程)。生成两个文件夹，metagraph保存生成的图，metadata保存与pattern同构的子图数量及*子图的位置*。所以需要先运行pattern_generator.py生成pattern后再运行metag_generator.py

pattern_checker.py:用来寻找是否有和pattern同构的子图。返回counting和subisomorphisms，前者统计同构数量，后者返回同构的子图。

用于寻找同构数量的代码在186行：counts += graph.count_subisomorphisms_vf2(pattern,
                        color1=vertex_colors[0], color2=vertex_colors[1],
                        edge_color1=edge_colors[0], edge_color2=edge_colors[1],
                        node_compat_fn=PatternChecker.node_compat_fn,
                        edge_compat_fn=PatternChecker.edge_compat_fn)
用于寻找同构子图的代码在156行：graph.get_subisomorphisms_vf2(pattern,
                    color1=vertex_colors[0], color2=vertex_colors[1],
                    edge_color1=edge_colors[0], edge_color2=edge_colors[1],
                    node_compat_fn=PatternChecker.node_compat_fn,
                    edge_compat_fn=PatternChecker.edge_compat_fn):

graph_generator.py:参数alpha的意义不是很清楚，但大致上的功能是用于调整子图大小从而加速寻找同构，总之应该不会影响生成数据集

run.py:生成graph，pattern，metadata（记录counting和subisomorphism），在上一级的data文件夹中
