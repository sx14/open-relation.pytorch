from open_relation.dataset.vrd.process.split_anno_pkg import split_anno_pkg
from open_relation.dataset.vrd.process.reformat_anno import reformat_anno
from open_relation.dataset.vrd.process import ext_pre_cnn_feat, ext_obj_cnn_feat
from open_relation.dataset.vrd.process.gen_label_weights import gen_label_weigths

if __name__ == '__main__':
    # split_anno_pkg()
    # reformat_anno()
    ext_obj_cnn_feat.gen_cnn_feat()
    ext_pre_cnn_feat.gen_cnn_feat()
    # gen_label_weigths('object')
    # gen_label_weigths('predicate')
