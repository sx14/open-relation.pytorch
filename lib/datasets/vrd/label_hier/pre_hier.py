import os
from lib.datasets.label_hier import LabelHier
from lib.datasets.label_hier import LabelNode

from global_config import PROJECT_ROOT


class PreNet(LabelHier):

    def _construct_hier(self):
        # root node
        # 0 is background
        next_label_ind = 1
        root = LabelNode('relation', next_label_ind, False)
        self._index2node.append(root)
        self._label2node['relation'] = root
        next_label_ind += 1

        # abstract level
        # interact, spatial, belong, comparison
        abs_level = {'interact.a': 'relation',
                     'spatial.a': 'relation',
                     'possess.a': 'relation',
                     'compare.a': 'relation'}


        # basic level
        basic_level = {
            'on.s': 'spatial.a',
            'wear.p': 'possess.a',
            'has.p': 'possess.a',
            'behind.s': 'spatial.a',
            'in the front of.s': 'spatial.a',
            'near.s': 'spatial.a',
            'under.s': 'spatial.a',
            'walk.i': 'interact.a',
            'in.s': 'spatial.a',
            'hold.i': 'interact.a',
            'with.p': 'possess.a',
            'carry.i': 'interact.a',
            'look.i': 'interact.a',
            'use.i': 'interact.a',
            'at.s': 'spatial.a',
            'attach to.p': 'possess.a',
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
        }


        sup_level = {
            'next to.s': 'near.s',
            'above.s': 'on.s',
            'over.s': 'on.s',
            'below.s': 'under.s',
            'beside.s': 'near.s',
            'rest on.s': 'on.s',
        }

        concrete_level = {
            'on': 'on.s',
            'wear': 'wear.p',
            'has': 'has.p',
            'next to': 'next to.s',
            'sleep next to': 'next to.s',
            'sit next to': 'next to.s',
            'stand next to': 'next to.s',
            'park next to': 'next to.s',
            'walk next to': 'next to.s',
            'above': 'above.s',
            'behind': 'behind.s',
            'stand behind': 'behind.s',
            'sit behind': 'behind.s',
            'park behind': 'behind.s',
            'in the front of': 'in the front of.s',
            'under': 'under.s',
            'stand under': 'under.s',
            'sit under': 'under.s',
            'near': 'near.s',
            'walk to': 'beside.s',
            'walk': 'walk.i',
            'walk past': 'beside.s',
            'in': 'in.s',
            'below': 'below.s',
            'beside': 'beside.s',
            'walk beside': 'beside.s',
            'over': 'over.s',
            'hold': 'hold.i',
            'by': 'beside.s',
            'beneath': 'on.s',
            'with': 'with.p',
            'on the top of': 'on.s',
            'on the left of': 'beside.s',
            'on the right of': 'beside.s',
            'sit on': 'on.s',
            'ride': 'on.s',
            'carry': 'carry.i',
            'look': 'look.i',
            'stand on': 'on.s',
            'use': 'use.i',
            'at': 'at.s',
            'attach to': 'attach to.p',
            'cover': 'cover.i',
            'touch': 'touch.i',
            'watch': 'look.i',
            'against': 'against.s',
            'inside': 'in.s',
            'adjacent to': 'next to.s',
            'across': 'across.s',
            'contain': 'contain.p',
            'drive': 'in.s',
            'drive on': 'on.s',
            'taller than': 'than.c',
            'eat': 'eat.i',
            'park on': 'on.s',
            'lying on': 'rest on.s',
            'pull': 'pull.i',
            'talk': 'talk.i',
            'lean on': 'on.s',
            'fly': 'fly.i',
            'face': 'face.i',
            'play with': 'play with.i',
            'sleep on': 'rest on.s',
            'outside of': 'outside of.s',
            'rest on': 'rest on.s',
            'follow': 'behind.s',
            'hit': 'hit.i',
            'feed': 'feed.i',
            'kick': 'kick.i',
            'skate on': 'on.s'
        }

        levels = [abs_level, basic_level, sup_level, concrete_level]
        for level in levels:
            for label in level:
                parent_label = level[label]
                parent_node = self._label2node[parent_label]
                assert parent_node is not None
                if label in concrete_level.keys():
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

# if __name__ == '__main__':
#     a = PreNet()
#     n = a.get_pre('stand next to')
#     n.show_hyper_paths()