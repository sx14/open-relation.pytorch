import os

from global_config import PROJECT_ROOT
from lib.datasets.label_hier import LabelHier
from lib.datasets.label_hier import LabelNode


class PreNet(LabelHier):

    def _construct_hier(self):
        # STUB IMPLEMENTATION

        # 0 is background
        # root node
        next_label_ind = 1
        root = LabelNode('relation.r', next_label_ind, False)
        self._index2node.append(root)
        self._label2node['relation.r'] = root
        next_label_ind += 1

        abstract_level = {
            'spacial.a': 'relation.r',
            'interact.a': 'relation.r',
            'possess.a': 'relation.r',
            'compare.c': 'relation.r'
        }

        basic_level = {
            'on.s': 'spacial.a',
            'across.s': 'spacial.a',
            'against.s': 'spacial.a',
            'near.s': 'spacial.a',
            'around.s': 'spacial.a',
            'behind.s': 'spacial.a',
            'at.s': 'spacial.a',
            'under.s': 'spacial.a',
            'between.s': 'spacial.a',
            'in.s': 'spacial.a',
            'in front of.s': 'spacial.a',
            'outside.s': 'spacial.a',

            'attach to.i': 'interact.a',
            'carry.i': 'interact.a',
            'carried by.i': 'interact.a',
            'cover.i': 'interact.a',
            'covered by.i': 'interact.a',
            'cross.i': 'interact.a',
            'cut.i': 'interact.a',
            'eat.i': 'interact.a',
            'face.i': 'interact.a',
            'contain.i': 'interact.a',
            'fly.i': 'interact.a',
            'hit.i': 'interact.a',
            'look.i': 'interact.a',
            'park.i': 'interact.a',
            'play.i': 'interact.a',
            'pull.i': 'interact.a',
            'read.i': 'interact.a',
            'ride.i': 'interact.a',
            'show.i': 'interact.a',
            'support.i': 'interact.a',
            'throw.i': 'interact.a',
            'touch.i': 'interact.a',
            'use.i': 'interact.a',
            'wear.i': 'interact.a',
            'wore by.i': 'interact.a',
            'walk.i': 'interact.a',

            'have.p': 'possess.a',
            'belong to.p': 'possess.a',

            'tall than.c': 'compare.c',
            'small than.c': 'compare.c',

        }

        supply_level = {
            'near': 'near.s',
        }

        supply1_level = {
            'on': 'on.s',
            'in': 'in.s',
            'of': 'belong to.p',
            'at': 'at.s',
            'behind': 'behind.s',
            'by': 'near',
            'next to': 'near',
            'hold': 'carry.i',
        }

        concrete_level = {
            'above': 'on.s',
            'across': 'across.s',
            'adorn': 'have.p',
            'against': 'against.s',
            'along': 'near',
            'around': 'around.s',
            'attach to': 'attach to.i',
            'belong to': 'belong to.p',
            'below': 'under.s',
            'beneath': 'under.s',
            'beside': 'near',
            'between': 'between.s',
            'build into': 'on.s',
            'carry': 'carry.i',
            'cast': 'have.p',
            'catch': 'carry.i',
            'connect to': 'attach to.i',
            'contain': 'contain.i',
            'cover': 'cover.i',
            'cover in': 'covered by.i',
            'cover with': 'covered by.i',
            'cross': 'cross.i',
            'cut': 'cut.i',
            'drive on': 'on',
            'eat': 'eat.i',
            'face': 'face.i',
            'fill with': 'contain.i',
            'fly': 'fly.i',
            'fly in': 'in',
            'for': 'belong to.p',
            'from': 'belong to.p',
            'grow in': 'in',
            'grow on': 'on',
            'hang in': 'in',
            'hang on': 'on',
            'have': 'have.p',
            'hit': 'hit.i',
            'hold by': 'carried by.i',
            'in front of': 'in front of.s',
            'in middle of': 'between.s',
            'inside': 'in.s',
            'lay in': 'in',
            'lay on': 'on',
            'lean on': 'on',
            'look at': 'look.i',
            'mount on': 'on',
            'on back of': 'on',
            'on bottom of': 'on',
            'on side of': 'near',
            'on top of': 'on',
            'outside': 'outside.s',
            'over': 'on.s',
            'paint on': 'on',
            'park': 'park.i',
            'part of': 'of',
            'play': 'play.i',
            'print on': 'on',
            'pull': 'pull.i',
            'read': 'read.i',
            'reflect in': 'in',
            'rest on': 'on',
            'ride': 'ride.i',
            'say': 'show.i',
            'show': 'show.i',
            'sit at': 'at',
            'sit in': 'in',
            'sit on': 'on',
            'small than': 'small than.c',
            'stand behind': 'behind',
            'stand on': 'on',
            'standing by': 'by',
            'standing in': 'in',
            'standing next to': 'next to',
            'support': 'support.i',
            'surround': 'around.s',
            'swing': 'hold',
            'tall than': 'tall than.c',
            'throw': 'throw.i',
            'to': 'belong to.p',
            'touch': 'touch.i',
            'under': 'under.s',
            'underneath': 'under.s',
            'use': 'use.i',
            'walk': 'walk.i',
            'walk in': 'in',
            'walk on': 'on',
            'watch': 'look.i',
            'wear': 'wear.i',
            'wear by': 'wore by.i',
            'with': 'have.p',
            'write on': 'on',
        }

        levels = [abstract_level, basic_level, supply_level, supply1_level, concrete_level]
        for level in levels:
            for label in level:
                parent_label = level[label]
                parent_node = self._label2node[parent_label]
                assert parent_node is not None
                if label in concrete_level.keys() or label in supply1_level.keys() or label in supply_level.keys():
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


label_path = os.path.join(PROJECT_ROOT, 'data', 'VGdevkit2007', 'VOC2007', 'predicate_labels.txt')
prenet = PreNet(label_path)

# raw_inds = prenet.get_raw_indexes()
# for ind in raw_inds:
#     n = prenet.get_node_by_index(ind)
#     n.show_hyper_paths()
#
# for ind in range(prenet.label_sum()):
#     node = prenet.get_node_by_index(ind)
#     cs = node.name() + ' : '
#     for c in node.children():
#         cs += c.name() + ', '
#
#     print(cs)
