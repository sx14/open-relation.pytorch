# coding: utf-8
from graphviz import Graph

from lib.datasets.vrd.label_hier.obj_hier import objnet
from lib.datasets.vrd.label_hier.pre_hier import prenet


def hill_sort(all_nodes):
    # curr = all_nodes[objnet.root().index()]
    # childrens = curr.children()
    des_nodes = sorted(all_nodes, key=lambda n: n.depth_ratio())
    acc_nodes = []
    for i in range(len(des_nodes)-1, 0, -2):
        node = des_nodes.pop(i)
        acc_nodes.append(node)
    return acc_nodes + des_nodes


labelnet = objnet

dot = Graph(comment='Label Hierarchy')
dot.attr(ratio='0.3')
# dot.node('a', 'a')
# dot.node('b', 'b')
# dot.edge('a', 'b')

all_nodes = []
for i in range(1, labelnet.label_sum()):
    all_nodes.append(labelnet.get_node_by_index(i))
hill_nodes = hill_sort(all_nodes)

for i in range(len(hill_nodes)):
    dot.node(str(hill_nodes[i].index()), hill_nodes[i].name().split('.')[0], shape='none')


for i in range(1, labelnet.label_sum()):
    n = labelnet.get_node_by_index(i)
    for c in n.children():
        dot.edge(str(n.index()), str(c.index()), shape='none')



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
