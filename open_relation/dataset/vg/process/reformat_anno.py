# -*- coding: utf-8 -*-
"""
step2 retain the objects and relationships with WordNet annotation
next: index_labels.py
"""
import os
import json
import nltk
from open_relation.dataset.dataset_config import DatasetConfig


legal_pos_tags = {
    # 名词
    'object': {u'NOUN': 'n',
               u'ADJ': None,
               u'VERB': None},
    # 动词，介词连词，名词（误），
    'predicate': {u'VERB': 'v',
                  u'ADP': 'v',
                  u'NOUN': 'v',
                  u'ADJ': None,
                  u'PRT': None,
                  u'ADV': None}
    # 'object': {
    #     'NN': None,
    #     'NNS': None,
    #     'NNP': None,
    #     'NNPS': None
    # },
    # 'predicate': {
    #     'IN': None,
    #     'JJR': None,
    #     'NN': 'v',
    #     'NNS': 'v',
    #     'RBR': None,
    #     'RP': None,
    #     'TO': None,
    #     'VB': 'v',
    #     'VBD': 'v',
    #     'VBG': 'v',
    #     'VBN': None,
    #     'VBP': 'v',
    #     'VBZ': 'v',
    # }
}


def regularize_obj_label(label):
    # type : object, predicate
    # pos_tags = legal_pos_tags['object']
    # tokens = nltk.word_tokenize(label.lower())
    # token_tags = nltk.pos_tag(tokens, tagset='universal')
    # legal_tokens = []
    #
    # for token_tag in token_tags:
    #     # reserve noun only
    #     if token_tag[1] in pos_tags:
    #         raw_token = token_tag[0]
    #         legal_tokens.append(raw_token)
    #     else:
    #         a = 1
    # if len(legal_tokens) == 0:
    #     legal_tokens.append(label)
    # return ' '.join(legal_tokens)
    return label.lower()


def regularize_pre_label(label, lemmatizer):
    pos_tags = legal_pos_tags['predicate']
    tokens = nltk.word_tokenize(label.lower())
    token_tags = nltk.pos_tag(tokens, tagset='universal')
    legal_tokens = []
    for token_tag in token_tags:
        # if token_tag[1] in pos_tags:
        #     if pos_tags[token_tag[1]] is not None:
        #         raw_token = lemmatizer.lemmatize(token_tag[0], pos=pos_tags[token_tag[1]])
        #     else:
        #         raw_token = token_tag[0]
        #         legal_tokens.append(raw_token)
        # else:
        #     a = 1
        raw_token = lemmatizer.lemmatize(token_tag[0], pos='v')
        legal_tokens.append(raw_token)

    if len(legal_tokens) == 0:
        raw_token = lemmatizer.lemmatize(label, pos='v')
        legal_tokens.append(raw_token)
    return ' '.join(legal_tokens)


def rlt_reformat(rlt_anno):

    def obj_reformat(obj_anno):
        obj = dict()
        obj['name'] = obj_anno['name']
        obj['ymin'] = int(obj_anno['y'])
        obj['ymax'] = int(obj_anno['y']+int(obj_anno['h']))
        obj['xmin'] = int(obj_anno['x'])
        obj['xmax'] = int(obj_anno['x']+int(obj_anno['w']))
        obj['synsets'] = obj_anno['synsets']
        obj['object_id'] = obj_anno['object_id']
        return obj

    sbj_anno = rlt_anno['subject']
    obj_anno = rlt_anno['object']
    sbj = obj_reformat(sbj_anno)
    obj = obj_reformat(obj_anno)
    pre = dict()
    pre['name'] = rlt_anno['predicate']
    pre['synsets'] = rlt_anno['synsets']
    # predicate box is union of obj box and sbj box
    pre['ymin'] = min(obj['ymin'], sbj['ymin'])
    pre['ymax'] = max(obj['ymax'], sbj['ymax'])
    pre['xmin'] = min(obj['xmin'], sbj['xmin'])
    pre['xmax'] = max(obj['xmax'], sbj['xmax'])
    new_rlt = dict()
    new_rlt['object'] = obj
    new_rlt['subject'] = sbj
    new_rlt['predicate'] = pre
    new_rlt['relationship_id'] = rlt_anno['relationship_id']
    return new_rlt


def wash_anno(dirty_anno_path, clean_anno_path):
    # use objects in relationships
    dirty_anno = json.load(open(dirty_anno_path, 'r'))
    clean_anno = dict()

    # extract unique objects from relationships
    id2obj = dict()
    id2rlt = dict()
    lemmatizer = nltk.WordNetLemmatizer()

    rlts = dirty_anno['relationships']
    for rlt in rlts:
        new_rlt = rlt_reformat(rlt)
        objs_have_synset = True
        objs = [new_rlt['subject'], new_rlt['object']]
        for obj in objs:
            if obj['object_id'] not in id2obj:
                if len(obj['synsets']) > 0:
                    # object must have wn synset
                    reg_label = regularize_obj_label(obj['name'])
                    # print('%s | %s' % (obj['name'], reg_label))
                    obj['name'] = reg_label
                    id2obj[obj['object_id']] = obj
                else:
                    objs_have_synset = False
        if objs_have_synset:
            reg_label = regularize_pre_label(new_rlt['predicate']['name'], lemmatizer)
            # print('%s | %s' % (new_rlt['predicate']['name'], reg_label))
            new_rlt['predicate']['name'] = reg_label
            id2rlt[new_rlt['relationship_id']] = new_rlt

    clean_anno['image_info'] = dirty_anno['image_info']
    clean_anno['objects'] = id2obj.values()
    clean_anno['relations'] = id2rlt.values()
    json.dump(clean_anno, open(clean_anno_path, 'w'), indent=4)


def reformat_anno():
    vg_config = DatasetConfig('vg')
    dirty_anno_root = vg_config.data_config['dirty_anno_root']
    clean_anno_root = vg_config.data_config['dirty_anno_root']
    anno_list = os.listdir(dirty_anno_root)
    anno_list = sorted(anno_list)
    anno_sum = len(anno_list)
    for i in range(0, anno_sum):
        print('processing wash_anno [%d/%d]' % (anno_sum, i+1))
        dirty_anno_path = os.path.join(dirty_anno_root, anno_list[i])
        clean_anno_path = os.path.join(clean_anno_root, anno_list[i])
        wash_anno(dirty_anno_path, clean_anno_path)


