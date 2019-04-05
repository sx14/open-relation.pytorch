# coding: utf-8
from graphviz import Graph

from lib.datasets.vrd.label_hier.obj_hier import objnet
from lib.datasets.vrd.label_hier.pre_hier import prenet

dot = Graph(comment='Label Hierarchy')
# dot.node('a', 'a')
# dot.node('b', 'b')
# dot.edge('a', 'b')

for i in range(1, objnet.label_sum()):
    dot.node(str(i), objnet.get_node_by_index(i).name())

for i in range(1, objnet.label_sum()):
    n = objnet.get_node_by_index(i)
    for c in n.children():
        dot.edge(str(n.index()), str(c.index()), shape='none')

# for i in range(1, prenet.label_sum()):
#     dot.node(str(-i), prenet.get_node_by_index(i).name())
#
# for i in range(1, prenet.label_sum()):
#     n = prenet.get_node_by_index(-i)
#     for c in n.children():
#         dot.edge(str(-n.index()), str(-c.index()))

# dot.view()


# 获取DOT source源码的字符串形式
print(dot.source)
# // The Test Table
# digraph {
#   A [label="Dot A"]
#   B [label="Dot B"]
#   C [label="Dot C"]
#   A -> B
#   A -> C
#   A -> B
#   B -> C [label=test]
# }


# 保存source到文件，并提供Graphviz引擎
dot.render('test-output/test-table.gv', view=True)
