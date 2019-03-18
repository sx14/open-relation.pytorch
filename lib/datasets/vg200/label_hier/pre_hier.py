import os
from lib.datasets.label_hier import LabelHier
from lib.datasets.label_hier import LabelNode
from global_config import PROJECT_ROOT


class PreNet(LabelHier):

    def _construct_hier(self):
        # STUB IMPLEMENTATION

        # 0 is background
        # root node
        next_label_ind = 1
        root = LabelNode('relation.r.01', next_label_ind, False)
        self._index2node.append(root)
        self._label2node['relation.r.01'] = root
        next_label_ind += 1

        basic_level = {

        }

        concrete_level = {
            'above': 'on.s',
            'across': '',
            'adorn': '',
            'against': '',
            'along': '',
            'around': '',
            'at': '',
            'attach to': '',
            'behind': '',
            'belong to': '',
            'below': '',
            'beneath': '',
            'beside': '',
            'between': '',
            'build into': '',
            'by': '',
            'carry': '',
            'cast': '',
            'catch': '',
            'connect to': '',
            'contain': '',
            'cover': '',
            'cover in': '',
            'cover with': '',
            'cross': '',
            'cut': '',
            'drive on': '',
            'eat': '',
            'face': '',
            'fill with': '',
            'fly': '',
            'fly in': '',
            'for': '',
            'from': '',
            'grow in': '',
            'grow on': '',
            'hang in': '',
            'hang on': '',
            'have': '',
            'hit': '',
            'hold': '',
            'hold by': '',
            'in': '',
            'in front of': '',
            'in middle of': '',
            'inside': '',
            'lay in': '',
            'lay on': '',
            'lean on': '',
            'look at': '',
            'mount on': '',
            'near': '',
            'next to': '',
            'of': '',
            'on': '',
            'on back of': '',
            'on bottom of': '',
            'on side of': '',
            'on top of': '',
            'outside': '',
            'over': '',
            'paint on': '',
            'park': '',
            'part of': '',
            'play': '',
            'print on': '',
            'pull': '',
            'read': '',
            'reflect in': '',
            'rest on': '',
            'ride': '',
            'say': '',
            'show': '',
            'sit at': '',
            'sit in': '',
            'sit on': '',
            'small than': '',
            'stand behind': '',
            'stand on': '',
            'standing by': '',
            'standing in': '',
            'standing next to': '',
            'support': '',
            'surround': '',
            'swing': '',
            'tall than': '',
            'throw': '',
            'to': '',
            'touch': '',
            'under': '',
            'underneath': '',
            'use': '',
            'walk': '',
            'walk in': '',
            'walk on': '',
            'watch': '',
            'wear': '',
            'wear by': '',
            'with': '',
            'write on': '',
        }


        # TODO: construct predicate label hierarchy

    def __init__(self, raw_label_path):
        LabelHier.__init__(self, raw_label_path)


label_path = os.path.join(PROJECT_ROOT, 'data', 'VGdevkit2007', 'VOC2007', 'predicate_labels.txt')
prenet = PreNet(label_path)
