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
    expanded_query = []
    for q in raw_query:
        sbj, pre, obj = q

        sbj_set = hier_expand(sbj, objnet)  # he, his children and his children's children, ...
        obj_set = hier_expand(obj, objnet)
        pre_set = hier_expand(pre, prenet)

        for pre in pre_set:
            opposite_pre = prenet.opposite(pre.name_prefix())
            for sbj in sbj_set:
                for obj in obj_set:
                    s = ' '.join([sbj.name_prefix(), pre.name_prefix(), obj.name_prefix()])
                    expanded_query.append(s)
                    if opposite_pre is not None:
                        s = ' '.join([obj.name_prefix(), opposite_pre, sbj.name_prefix()])
                        expanded_query.append(s)
    return expanded_query
