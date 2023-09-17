#%%
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from os.path import join
import utils
import importlib
import faiss
importlib.reload(utils)

# --------------------------------------------------------------------------------------------------------------------- #
NETWORK = 'contrast'
# Choose NETWORK from |'teacher_triplet'|'student_contrast'|'student_triplet'|'student_quadruplet'|
# --------------------------------------------------------------------------------------------------------------------- #

NUM_BINS = 11
SHOW_AP = True
exp = NETWORK
resume = join('logs', NETWORK)

with open(join(resume, 'stu_30k.pickle'), 'rb') as handle:
    q_mu = pickle.load(handle)
    db_mu = pickle.load(handle)
    q_sigma_sq = pickle.load(handle)
    db_sigma_sq = pickle.load(handle)
    preds = pickle.load(handle)
    dists = pickle.load(handle)
    gt = pickle.load(handle)
    _ = pickle.load(handle)
    _ = pickle.load(handle)



q_sigma_sq_h = np.mean(q_sigma_sq, axis=1)
db_sigma_sq_h = np.mean(db_sigma_sq, axis=1)
indices, _, _ = utils.get_zoomed_bins(q_sigma_sq_h, NUM_BINS)

n_values = [1, 5, 10]

#print(preds.shape)
#print(gt)
# ---------------------------- recognition metric ---------------------------- #
recall = utils.cal_recall(preds, gt, n_values) / 100.0
print('rec@1/5/10: {:.3f} / {:.3f} / {:.3f}'.format(recall[0], recall[1], recall[2]))
map = [utils.cal_mapk(preds, gt, n) / 100.0 for n in n_values]
print('mAP@1/5/10: {:.3f} / {:.3f} / {:.3f}'.format(map[0], map[1], map[2]))

if SHOW_AP:
    recalls, precisions = utils.bin_pr(preds, dists, gt)
    ap = 0
    for index_j in range(len(recalls) - 1):
        ap += precisions[index_j] * (recalls[index_j + 1] - recalls[index_j])

    print('AP: {:.3f}'.format(ap))



