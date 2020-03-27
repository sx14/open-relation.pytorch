from app.database.rela_db import RelaDB
import pickle

rela_db = RelaDB('rela_db.db')
dataset = 'vg'

'''
class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


pred_roidb_path = '../../../hier_rela/rela_box_label_%s_hier_pure.bin' % (dataset)
pred_roidb = pickle.load(StrToBytes(open(pred_roidb_path)), encoding='bytes')
print('image db loaded')

for img_id in pred_roidb:
    pr_curr = pred_roidb[img_id]
    rela_db.insert_rela([(int(rela[1]), int(rela[0]), int(rela[2]), img_id) for rela in pr_curr])
'''
print(rela_db.stat_rela())
rela_db.close()
