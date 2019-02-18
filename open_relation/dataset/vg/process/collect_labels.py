import os
import json
import matplotlib.pyplot as plt
from open_relation.dataset.dataset_config import DatasetConfig


vg_config = DatasetConfig('vg')


def count():
    # counter
    obj_counter = dict()
    pre_counter = dict()
    obj2wn = dict()
    pre2wn = dict()

    # counting
    clean_anno_root = vg_config.data_config['dirty_anno_root']
    anno_list = os.listdir(clean_anno_root)
    anno_num = len(anno_list)
    for i, anno_name in enumerate(anno_list):
        print('counting [%d/%d]' % (anno_num, i+1))
        anno_path = os.path.join(clean_anno_root, anno_name)
        anno = json.load(open(anno_path, 'r'))
        objs = anno['objects']
        for obj in objs:
            synsets = set(obj['synsets'])
            name = obj['name']
            if name in obj_counter:
                obj_counter[name] += 1
            else:
                obj_counter[name] = 1
            if name in obj2wn:
                obj2wn[name] = obj2wn[name] | synsets
            else:
                obj2wn[name] = synsets

        relations = anno['relations']
        for rlt in relations:
            synsets = set(rlt['predicate']['synsets'])
            predicate = rlt['predicate']['name']
            if predicate in pre_counter:
                pre_counter[predicate] += 1
            else:
                pre_counter[predicate] = 1
            if predicate in pre2wn:
                pre2wn[predicate] = pre2wn[predicate] | synsets
            else:
                pre2wn[predicate] = synsets


    counters = {
        'object': (obj_counter, obj2wn, 1000),
        'predicate': (pre_counter, pre2wn, 500)
    }

    return counters


def collect_labels():

    counters = count()

    for target in counters:
        counter, raw2wn, top = counters[target]
        label_list = []
        raw_label_list = []
        sorted_count = sorted(counter.items(), key=lambda a: a[1])
        sorted_count.reverse()

        for i, (name, c) in enumerate(sorted_count):
            # retain top N
            if i < top:
                raw_label_list.append(name+'\n')
                line = name + '|'
                syns = raw2wn[name]
                for syn in syns:
                    line = line + syn + ' '
                line = line.strip(' ')
                label_list.append(line+'\n')
                print('%d %s: %d' % (i + 1, name, c))
            else:
                break

        # save label list
        raw_label_list_path = os.path.join(vg_config.dataset_root, target+'_labels.txt')
        with open(raw_label_list_path, 'w') as f:
            f.writelines(raw_label_list)

        label_list_path = os.path.join(vg_config.dataset_root, target+'_label2wn.txt')
        with open(label_list_path, 'w') as f:
            f.writelines(label_list)

        # counts = [item[1] for item in sorted_count]
        # plt.plot(range(len(label_list)), counts[:len(label_list)])
        # plt.title('distribution')
        # plt.xlabel('object')
        # plt.ylabel('count')
        # plt.show()


