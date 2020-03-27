from global_config import VG_ROOT, VRD_ROOT

dataset = 'vg'

if dataset == 'vrd':
    ds_root = VRD_ROOT
    from lib.datasets.vrd.label_hier.obj_hier import objnet
    from lib.datasets.vrd.label_hier.pre_hier import prenet
else:
    ds_root = VG_ROOT
    from lib.datasets.vg200.label_hier.obj_hier import objnet
    from lib.datasets.vg200.label_hier.pre_hier import prenet


def hier_expand(label, labelnet):
    return labelnet.get_node_by_name_prefix(label).descendants()


def expand_query(raw_query):
    """
    expand the query by permuting and combining the hyponymy.
    exchange sbj and obj if predicate has antonym.
    """
    for q in raw_query:
        sbj, pre, obj = q
        sbj_set = hier_expand(sbj, objnet)  # he, his children and his children's children, ...
        obj_set = hier_expand(obj, objnet)
        pre_set = hier_expand(pre, prenet)

    return [s.index() for s in sbj_set], [p.index() for p in pre_set], [o.index() for o in obj_set]
