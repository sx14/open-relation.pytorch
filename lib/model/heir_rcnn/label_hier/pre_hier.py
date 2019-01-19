import os
from label_hier import LabelHier
from label_hier import LabelNode
from lib.datasets.vrd import path_config

class PreNet(LabelHier):

    def _construct_hier(self):
        # root node
        # 0 is background
        next_label_ind = 1
        root = LabelNode('predicate', next_label_ind)
        self._index2node.append(root)
        self._label2node['predicate'] = root
        next_label_ind += 1

        # abstract level
        # action, spatial, association, comparison
        abs_labels = ['act', 'spa', 'ass', 'cmp']
        for abs_label in abs_labels:
            node = LabelNode(abs_label, next_label_ind)
            self._index2node.append(node)
            next_label_ind += 1
            node.append_hyper(root)
            self._label2node[abs_label] = node

        # basic level
        act_labels = ['wear',    'sleep',    'sit',      'stand',
                        'park',    'walk',     'hold',     'ride',
                        'carry',   'look',     'use',      'cover',
                        'touch',   'watch',    'drive',    'eat',
                        'lying',   'pull',     'talk',     'lean',
                        'fly',     'face',     'rest',     'skate',
                        'follow',  'hit',      'feed',     'kick',
                        'play with']

        spa_labels = ['on',      'next to',  'above',    'behind',
                        'under',   'near',     'in',       'below',
                        'beside',  'over',     'by',       'beneath',
                        'on the top of',        'in the front of',
                        'on the left of',       'on the right of',
                        'at',      'against',  'inside',   'adjacent to',
                        'across',  'outside of' ]

        ass_labels = ['has',     'with',     'attach to',    'contain']

        cmp_labels = ['than']

        basic_label_lists = [act_labels, spa_labels, ass_labels, cmp_labels]
        for i in range(len(abs_labels)):
            abs_label = abs_labels[i]
            abs_pre = self._label2node[abs_label]
            basic_labels = basic_label_lists[i]
            # link basic level to abstract level
            for basic_label in basic_labels:
                node = LabelNode(basic_label, next_label_ind)
                next_label_ind += 1
                self._index2node.append(node)
                node.append_hyper(abs_pre)
                self._label2node[basic_label] = node

        # concrete level
        for raw_label in self._raw_labels:
            if raw_label not in self._label2node:
                # predicate phrase
                node = LabelNode(raw_label, next_label_ind)
                next_label_ind += 1
                self._index2node.append(node)
                first_space_pos = raw_label.find(' ')
                if first_space_pos == -1:
                    # print('<%s> Not a phrase !!!' % raw_pre)
                    exit(-1)
                phrase = [raw_label[:first_space_pos], raw_label[first_space_pos+1:]]
                for part in phrase:
                    if part in self._label2node:
                        hyper = self._label2node[part]
                        node.append_hyper(hyper)
                    else:
                        # print(' <%s> -> <%s> miss' % (raw_pre, part))
                        pass
                self._label2node[raw_label] = node

    def __init__(self, pre_label_path):
        LabelHier.__init__(self, pre_label_path)


label_path = os.path.join(path_config.vrd_root, 'predicate_labels.txt')
prenet = PreNet(label_path)

# if __name__ == '__main__':
#     a = PreNet()
#     n = a.get_pre('stand next to')
#     n.show_hyper_paths()