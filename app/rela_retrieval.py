import time

from database.rela_db import RelaDB
from lib.datasets.vg200.label_hier.obj_hier import objnet
from lib.datasets.vg200.label_hier.pre_hier import prenet
from utils.query_expansion import hier_expand

rela_db = RelaDB('./database/scripts/rela_db.db')


def show_rela(pr_curr):
    print('=====')
    for i in range(len(pr_curr)):
        pr_cls = pr_curr[i][1]
        obj_cls = pr_curr[i][2]
        sbj_cls = pr_curr[i][0]
        print('%s %s %s %d' % (
            objnet.get_node_by_index(int(sbj_cls)).name_prefix(), prenet.get_node_by_index(int(pr_cls)).name_prefix(),
            objnet.get_node_by_index(int(obj_cls)).name_prefix(),

            pr_curr[i][3]))


def search_by_concept(concept):
    search_node = objnet.get_node_by_name_prefix(concept)
    if search_node is None:
        return None
    start_tic = time.time()
    expanded_search_concepts = [node.index() for node in hier_expand(concept, objnet)]
    res = rela_db.stat_rela(expanded_search_concepts)
    end_tic = time.time()
    print('search time: %ss' % round(end_tic - start_tic, 2))
    return {
        'hypernym': search_node.hypers()[0].name_prefix(),
        'hyponyms': [node.name_prefix() for node in search_node.children()],
        'relas': [[prenet.get_node_by_index(t[0]).name_prefix(), objnet.get_node_by_index(t[1]).name_prefix(), t[2]] for
                  t in res],
        'concept': concept
    }
