import os
import pickle
from nltk.corpus import wordnet as wn
from open_relation.dataset.label_hier import LabelHier
from open_relation.dataset.label_hier import LabelNode
from open_relation.dataset.dataset_config import DatasetConfig

class ObjNet(LabelHier):


    def _raw_to_wn(self):
        raw2wn = dict()
        for vrd_label in self._raw_labels:
            vrd_label = vrd_label.strip()
            syns = wn.synsets(vrd_label)
            if len(syns) > 0:
                raw2wn[vrd_label] = [syns[0].name()]
            else:
                raw2wn[vrd_label] = ['']
        # fix auto annotation
        raw2wn['shoes'] = ['shoe.n.01']
        raw2wn['bike'] = ['bicycle.n.01']
        raw2wn['plate'] = ['plate.n.04']
        raw2wn['trash can'] = ['ashcan.n.01']
        raw2wn['traffic light'] = ['traffic_light.n.01']
        raw2wn['truck'] = ['truck.n.01']
        raw2wn['van'] = ['van.n.05']
        raw2wn['mouse'] = ['mouse.n.04']
        raw2wn['hydrant'] = ['fireplug.n.01']
        raw2wn['pants'] = ['trouser.n.01']
        raw2wn['jeans'] = ['trouser.n.01']
        raw2wn['monitor'] = ['monitor.n.04']
        raw2wn['post'] = ['post.n.04']
        return raw2wn

    def _create_label_nodes(self, raw2wn):
        path_choice = {
            'person': 1,
            'truck': 1,
            'shirt': 1,
            'car': 1,
            'hat': 1,
            'pants': 1,
            'motorcycle': 1,
            'jacket': 1,
            'bike': 1,
            'coat': 1,
            'dog': 0,
            'skateboard': 0,
            'cup': 1,
            'van': 1,
            'shorts': 1,
            'jeans': 1,
            'elephant': 0,
            'sand': 1,
            'cart': 1,
            'pot': 1,
            'paper': 1,
            'tie': 1,
        }

        next_label_index = 1
        # except 'background'
        for raw_label in self._raw_labels[1:]:
            wn_label = raw2wn[raw_label][0]
            wn_node = wn.synset(wn_label)
            hypernym_paths = wn_node.hypernym_paths()   # including wn_node self
            if len(hypernym_paths) > 1:
                hypernym_path = hypernym_paths[path_choice[raw_label]]
            else:
                hypernym_path = hypernym_paths[0]
            for i, w in enumerate(hypernym_path):
                node = self.get_node_by_name(w.name())
                if node is None:
                    node = LabelNode(w.name(), next_label_index, False)
                    self._label2node[w.name()] = node
                    self._index2node.append(node)
                    next_label_index += 1
                if i > 0:
                    node.add_hyper(last_node)
                    last_node.add_child(node)
                last_node = node

            # raw label is unique
            raw_node = LabelNode(raw_label, next_label_index, True)
            self._label2node[raw_label] = raw_node
            self._index2node.append(raw_node)
            next_label_index += 1
            wn_node = self.get_node_by_name(raw2wn[raw_label][0])
            raw_node.add_hyper(wn_node)
            wn_node.add_child(raw_node)

    def _construct_hier(self):
        raw2wn = self._raw_to_wn()
        self._create_label_nodes(raw2wn)

    def _fill_weights(self, weight_path):
        if os.path.exists(weight_path):
            ind2weight = pickle.load(open(weight_path, 'rb'))
            for ind in range(len(ind2weight)):
                node = self.get_node_by_index(ind)
                node.set_weight(ind2weight[ind])
        else:
            print('ObjNet: ind2weight not exists.')

    def __init__(self, raw_label_path, weight_path):
        LabelHier.__init__(self, raw_label_path)
        self._fill_weights(weight_path)


dataset_config = DatasetConfig('vrd')
label_path = os.path.join(dataset_config.dataset_root, 'object_labels.txt')
weight_path = dataset_config.extra_config['object'].config['ind2weight_path']
objnet = ObjNet(label_path, weight_path)
# raw2path = objnet.raw2path()
# if __name__ == '__main__':
#     a = ObjNet(label_path)
#     n = a.get_node_by_name('road')
#     n.show_hyper_paths()