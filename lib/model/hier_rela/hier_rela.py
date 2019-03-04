from torch import nn

class HierRela(nn.Module):
    def __init__(self, hierVis, hierLang):
        super(HierRela, self).__init__()
        self._hierVis = hierVis
        self._hierLang = hierLang


    def forward(self, im_data, im_info, gt_relas, num_relas):
        rois, cls_score, \
        RCNN_loss_cls, rois_label = self._hierVis(im_data, im_info, gt_relas, num_relas)

        rois_label_np = rois_label.data
        sbj_labels = rois_label_np[:, 0]
        pre_labels = rois_label_np[:, 1]
        obj_labels = rois_label_np[:, 2]

