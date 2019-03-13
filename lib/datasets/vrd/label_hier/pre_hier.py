import os
from lib.datasets.label_hier import LabelHier
from lib.datasets.label_hier import LabelNode

from global_config import PROJECT_ROOT


class PreNet(LabelHier):

    def pos_leaf_sum(self):
        # overwrite

        def dfs_count(node):
            if len(node.children()) == 0:
                return 1
            else:
                count = 0
                for c in node.children():
                    count += dfs_count(c)
                return count

        root = self.root()
        return dfs_count(root)

    def _construct_hier(self):
        # root node
        # 0 is background
        next_label_ind = 1
        root = LabelNode('relation.r.01', next_label_ind, False)
        self._index2node.append(root)
        self._label2node['relation.r.01'] = root
        next_label_ind += 1

        # abstract level
        # interact, spatial, belong, comparison
        abs_level = {'interact.a': 'relation.r.01',
                     'spatial.a': 'relation.r.01',
                     'possess.a': 'relation.r.01',
                     'compare.a': 'relation.r.01'}


        # basic level
        basic_level = {
            'on.s': 'spatial.a',
            'wear.i': 'interact.a',
            'has.p': 'possess.a',
            'behind.s': 'spatial.a',
            'in the front of.s': 'spatial.a',
            'near.s': 'spatial.a',
            'under.s': 'spatial.a',
            'walk.i': 'interact.a',
            'in.s': 'spatial.a',
            'with.p': 'possess.a',
            'carry.i': 'interact.a',
            'look.i': 'interact.a',
            'use.i': 'interact.a',
            'at.s': 'spatial.a',
            'attach to.i': 'interact.a',
            'touch.i': 'interact.a',
            'against.s': 'spatial.a',
            'across.s': 'spatial.a',
            'contain.p': 'possess.a',
            'than.c': 'compare.a',
            'eat.i': 'interact.a',
            'pull.i': 'interact.a',
            'talk.i': 'interact.a',
            'fly.i': 'interact.a',
            'face.i': 'interact.a',
            'play with.i': 'interact.a',
            'outside of.s': 'spatial.a',
            'hit.i': 'interact.a',
            'feed.i': 'interact.a',
            'kick.i': 'interact.a',
            'cover.i': 'interact.a',
            'drive.i': 'interact.a',
            'ride.i': 'interact.a',
        }

        sup_level = {
            'near': 'near.s',
            'behind': 'behind.s',
            'under': 'under.s',
            'on': 'on.s',
        }

        sup_level1 = {
            'beside': 'near',
            'next to': 'near',
            'rest on': 'on',
        }

        concrete_level = {
            'wear': 'wear.i',
            'has': 'has.p',
            'sleep next to': 'next to',
            'sit next to': 'next to',
            'stand next to': 'next to',
            'park next to': 'next to',
            'walk next to': 'next to',
            'above': 'on.s',
            'stand behind': 'behind',
            'sit behind': 'behind',
            'park behind': 'behind',
            'in the front of': 'in the front of.s',
            'stand under': 'under',
            'sit under': 'under',
            'walk to': 'walk.i',
            'walk': 'walk.i',
            'walk past': 'walk.i',
            'in': 'in.s',
            'below': 'under',
            'walk beside': 'beside',
            'over': 'on.s',
            'hold': 'carry.i',
            'by': 'beside',
            'beneath': 'under',
            'with': 'with.p',
            'on the top of': 'on',
            'on the left of': 'beside',
            'on the right of': 'beside',
            'sit on': 'on',
            'ride': 'ride.i',
            'carry': 'carry.i',
            'look': 'look.i',
            'stand on': 'on',
            'use': 'use.i',
            'at': 'at.s',
            'attach to': 'attach to.i',
            'cover': 'cover.i',
            'touch': 'touch.i',
            'watch': 'look.i',
            'against': 'against.s',
            'inside': 'in.s',
            'adjacent to': 'next to',
            'across': 'across.s',
            'contain': 'contain.p',
            'drive': 'drive.i',
            'drive on': 'on',
            'taller than': 'than.c',
            'eat': 'eat.i',
            'park on': 'on',
            'lying on': 'rest on',
            'pull': 'pull.i',
            'talk': 'talk.i',
            'lean on': 'on',
            'fly': 'fly.i',
            'face': 'face.i',
            'play with': 'play with.i',
            'sleep on': 'rest on',
            'outside of': 'outside of.s',
            'follow': 'behind',
            'hit': 'hit.i',
            'feed': 'feed.i',
            'kick': 'kick.i',
            'skate on': 'on'
        }

        levels = [abs_level, basic_level, sup_level, sup_level1, concrete_level]
        for level in levels:
            for label in level:
                parent_label = level[label]
                parent_node = self._label2node[parent_label]
                assert parent_node is not None
                if label in concrete_level.keys() or label in sup_level.keys() or label in sup_level1.keys():
                    node = LabelNode(label, next_label_ind, True)
                else:
                    node = LabelNode(label, next_label_ind, False)
                self._index2node.append(node)
                self._label2node[label] = node
                node.add_hyper(parent_node)
                parent_node.add_child(node)
                next_label_ind += 1

    def __init__(self, raw_label_path):
        LabelHier.__init__(self, raw_label_path)


label_path = os.path.join(PROJECT_ROOT, 'data', 'VRDdevkit2007', 'VOC2007', 'predicate_labels.txt')
prenet = PreNet(label_path)

# for i in prenet.get_raw_indexes():
#     n = prenet.get_node_by_index(i)
#     n.show_hyper_paths()
#
# for i in range(prenet.label_sum()):
#     n = prenet.get_node_by_index(i)
#     cs = ''
#     for c in n.children():
#         cs = cs + ' | ' + c.name()
#     print(n.name()+ ':' + cs)