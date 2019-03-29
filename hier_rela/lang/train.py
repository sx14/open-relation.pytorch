import os
import shutil

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy as loss_func
from tensorboardX import SummaryWriter

from lang_dataset import LangDataset
from lang_config import train_params, data_config
from lib.model.hier_rela.lang.hier_lang import HierLang
from lib.model.hier_rela.lang.hier_lang import order_softmax_test as rank_test
from global_config import HierLabelConfig


def eval(model, test_dl):
    model.eval()
    acc_sum = 0
    loss_sum = 0
    batch_num = 0
    for batch in test_dl:
        batch_num += 1
        sbj1, pre1, obj1, pos_neg_inds = batch
        v_sbj1 = Variable(sbj1).float().cuda()
        v_obj1 = Variable(obj1).float().cuda()
        with torch.no_grad():
            pre_scores1 = model(v_sbj1, v_obj1)

        acc, loss_scores, y = rank_test(pre_scores1, pos_neg_inds)
        loss = loss_func(loss_scores, y)
        acc_sum += acc
        loss_sum += loss
    avg_acc = acc_sum / batch_num
    avg_loss = loss_sum / batch_num
    model.train()
    return avg_acc, avg_loss


""" ======= train ======= """

dataset = 'vrd'
if dataset == 'vrd':
    from lib.datasets.vrd.label_hier.pre_hier import prenet
else:
    from lib.datasets.vg200.label_hier.pre_hier import prenet

# training hyper params
lr = train_params['lr']
epoch_num = train_params['epoch_num']
batch_size = train_params['batch_size']

# init dataset
obj_config = HierLabelConfig(dataset, 'object')
pre_config = HierLabelConfig(dataset, 'predicate')
obj_label_vec_path = obj_config.label_vec_path()
pre_label_vec_path = pre_config.label_vec_path()
rlt_path = data_config['train']['raw_rlt_path']+dataset
train_set = LangDataset(rlt_path, obj_label_vec_path, pre_label_vec_path, prenet)
train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)

rlt_path = data_config['test']['raw_rlt_path']+dataset
test_set = LangDataset(rlt_path, obj_label_vec_path, pre_label_vec_path, prenet)
test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# model
save_model_root = 'output/%s/' % dataset
if not os.path.isdir(save_model_root):
    os.makedirs(save_model_root)

new_model_path = os.path.join(save_model_root, train_params['latest_model_path']+dataset+'.pth')
best_model_path = os.path.join(save_model_root, train_params['best_model_path']+dataset+'.pth')

input_length = train_set.obj_vec_length() * 2
gt_label_vec_path = pre_label_vec_path
model = HierLang(input_length, gt_label_vec_path)
if os.path.exists(new_model_path):
    model.load_state_dict(torch.load(new_model_path))
    print('Loading weights success.')
else:
    print('No pretrained weights.')
model.cuda()

# optimizer

weight_p, bias_p = [], []
for name, p in model.named_parameters():
    if 'bias' in name:
        bias_p += [p]
    else:
        weight_p += [p]
optim = torch.optim.SGD([{'params': weight_p, 'weight_decay': 1e-5},
                         {'params': bias_p, 'weight_decay': 0}], lr=lr)

# training process record
if os.path.exists('logs'):
    shutil.rmtree('logs')
sw = SummaryWriter('logs')
batch_num = 0
best_acc = 0

for epoch in range(epoch_num):
    for batch in train_dl:
        batch_num += 1
        sbj1, pre1, obj1, pos_neg_inds = batch
        v_sbj1 = Variable(sbj1).float().cuda()
        v_pre1 = Variable(pre1).float().cuda()
        v_obj1 = Variable(obj1).float().cuda()

        pre_scores1 = model(v_sbj1, v_obj1)

        acc, loss_scores, y = rank_test(pre_scores1, pos_neg_inds)
        loss = loss_func(loss_scores, y)
        sw.add_scalars('acc', {'train': acc}, batch_num)
        sw.add_scalars('loss', {'train': loss}, batch_num)

        print('Epoch %d | Batch %d | Loss %.2f | Acc: %.2f' % (epoch + 1, batch_num, loss.cpu().data, acc))

        optim.zero_grad()
        loss.backward()
        optim.step()

    print('\nevaluating ......')
    avg_acc, avg_loss = eval(model, test_dl)
    sw.add_scalars('acc', {'eval': avg_acc}, batch_num)
    sw.add_scalars('loss', {'eval': avg_loss}, batch_num)
    if avg_acc > best_acc:
        best_acc = avg_acc
        torch.save(model.state_dict(), best_model_path)
    print('>>>> Eval Acc: % .2f <<<<\n' % avg_acc)
    torch.save(model.state_dict(), new_model_path)


