import os
from open_relation.dataset.label_hier import LabelHier
from open_relation.dataset.label_hier import LabelNode
from open_relation.dataset.dataset_config import DatasetConfig

class PreNet(LabelHier):

    def _construct_hier(self):
        # root node
        # 0 is background
        next_label_ind = 1
        root = LabelNode('predicate', next_label_ind, False)
        self._index2node.append(root)
        self._label2node['predicate'] = root
        next_label_ind += 1

        # abstract level
        # action, spatial, association, comparison
        abs_labels = ['act', 'spa', 'ass', 'cmp']
        for abs_label in abs_labels:
            node = LabelNode(abs_label, next_label_ind, False)
            self._index2node.append(node)
            next_label_ind += 1
            node.add_hyper(root)
            self._label2node[abs_label] = node

        # basic level
        act_labels = [  'kick',     'sit',      'stand',
                        'park',     'walk',     'hold',      'ride',
                        'carry',    'look',     'use',       'cover',
                        'touch',    'watch',    'drive',     'eat',
                        'pull',     'talk',     'lean',
                        'fly',      'face',     'rest',      'skate',
                        'follow',   'hit',      'feed',      'play with']

        spa_labels = [  'on.s',     'under.s',
                        'near.s',     'in.s',
                        'in the front of',  'behind',
                        'at',      'against',
                        'across',  'outside of']

        ass_labels = [  'has',     'with.a',     'attach to',    'contain']

        cmp_labels = [  'than']


        basic_label_lists = [act_labels, spa_labels, ass_labels, cmp_labels]
        for i in range(len(abs_labels)):
            abs_label = abs_labels[i]
            abs_pre = self._label2node[abs_label]
            basic_labels = basic_label_lists[i]
            # link basic level to abstract level
            for basic_label in basic_labels:
                node = LabelNode(basic_label, next_label_ind, False)
                next_label_ind += 1
                self._index2node.append(node)
                node.add_hyper(abs_pre)
                self._label2node[basic_label] = node

        # supply basic level
        supplement = [
            # action
            ('sleep', 'rest'),
            ('lying', 'rest'),
            # spatial
            ('on', 'on.s'),
            ('above', 'on.s'),
            ('over', 'on.s'),
            ('on the top of', 'on.s'),
            ('under', 'under.s'),
            ('below', 'under.s'),
            ('beneath', 'under.s'),
            ('in', 'in.s'),
            ('inside', 'in.s'),
            ('near', 'near.s'),
            ('beside.s', 'near.s'),
            ('beside', 'beside.s'),
            ('by', 'beside.s'),
            ('on the left of', 'beside.s'),
            ('on the right of', 'beside.s'),
            ('next to', 'beside.s'),
            ('adjacent to', 'beside.s'),
            # association
            ('with', 'with.a'),
            ('wear', 'with.a'),
        ]

        for s in supplement:
            child_label = s[0]
            parent_label = s[1]
            if parent_label not in self._label2node:
                print(parent_label + ' not create !')
                exit(-1)
            parent_node = self._label2node[parent_label]
            child_node = LabelNode(child_label, next_label_ind, False)
            next_label_ind += 1
            self._index2node.append(child_node)
            self._label2node[child_label] = child_node
            child_node.add_hyper(parent_node)

        # concrete level
        for raw_label in self._raw_labels:
            if raw_label not in self._label2node:
                # predicate phrase
                node = LabelNode(raw_label, next_label_ind, True)
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
                        node.add_hyper(hyper)
                    else:
                        # print(' <%s> -> <%s> miss' % (raw_label, part))
                        pass
                self._label2node[raw_label] = node
            else:
                raw_node = self._label2node[raw_label]
                raw_node.set_raw(True)

    def __init__(self, raw_label_path):
        LabelHier.__init__(self, raw_label_path)


dataset_config = DatasetConfig('vrd')
label_path = os.path.join(dataset_config.dataset_root, 'predicate_labels.txt')
prenet = PreNet(label_path)

# if __name__ == '__main__':
#     a = PreNet()
#     n = a.get_pre('stand next to')
#     n.show_hyper_paths()