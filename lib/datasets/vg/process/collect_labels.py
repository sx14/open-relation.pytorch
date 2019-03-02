import os
import json
import matplotlib.pyplot as plt


def count(vg_config):
    # counter
    obj_counter = dict()
    pre_counter = dict()
    obj2wn = dict()
    pre2wn = dict()

    # counting
    clean_anno_root = vg_config['dirty_anno_root']
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

        relations = anno['relationships']
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
        'object': (obj_counter, obj2wn, 2000),
        'predicate': (pre_counter, pre2wn, 1000)
    }

    return counters


def collect_labels(vg_config):

    counters = count(vg_config)

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
        raw_label_list_path = vg_config['_raw_label_path' % target[:3]]
        with open(raw_label_list_path, 'w') as f:
            f.writelines(raw_label_list)

        label_list_path = vg_config['_raw2wn_path' % target[:3]]
        with open(label_list_path, 'w') as f:
            f.writelines(label_list)

        # counts = [item[1] for item in sorted_count]
        # plt.plot(range(len(label_list)), counts[:len(label_list)])
        # plt.title('distribution')
        # plt.xlabel('object')
        # plt.ylabel('count')
        # plt.show()


