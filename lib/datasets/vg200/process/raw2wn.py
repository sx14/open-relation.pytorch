import os
from nltk.corpus import wordnet as wn


def raw2wn(raw_label_path, raw2wn_path):
    if not os.path.exists(raw_label_path):
        print(raw_label_path + ' not exists.')
        exit(-1)

    with open(raw_label_path, 'r') as f:
        raw_labels = f.readlines()
    for i in range(len(raw_labels)):
        raw_labels[i] = raw_labels[i].strip()
        wn_syns = wn.synsets(raw_labels[i])
        if len(wn_syns) > 0:
            raw_labels[i] = raw_labels[i] + '|' + wn_syns[0].name()+'\n'
        else:
            raw_labels[i] = raw_labels[i] + '\n'

    with open(raw2wn_path+'.gen', 'w') as f:
        f.writelines(raw_labels)
