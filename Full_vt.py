from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import no_grad
import numpy as np

def full_vt(epoch, model, data, prefix, writer=None):   
    print(prefix+' start...')
    model.eval()

    with no_grad():
        precision, recall, ndcg_score = model.full_accuracy(data)
        print('---------------------------------{0}-th Precition:{1:.4f} Recall:{2:.4f} NDCG:{3:.4f}---------------------------------'.format(
            epoch, precision, recall, ndcg_score))

        # writer.add_scalar(prefix+'_Precition', precision, epoch)
        # writer.add_scalar(prefix+'_Recall', recall, epoch)
        # writer.add_scalar(prefix+'_NDCG', ndcg_score, epoch)

        # writer.add_histogram(prefix+'_visual_distribution', model.v_rep, epoch)
        # writer.add_histogram(prefix+'_acoustic_distribution', model.a_rep, epoch)
        # writer.add_histogram(prefix+'_textual_distribution', model.t_rep, epoch)

        return precision, recall, ndcg_score
def full_t(epoch, model, prefix,writer=None):
    print(prefix + ' start...')
    model.eval()

    with no_grad():
        precision, recall, ndcg_score = model.accuracy()
        print(
            '---------------------------------{0}-th Precition:{1:.4f} Recall:{2:.4f} NDCG:{3:.4f}---------------------------------'.format(
                epoch, precision, recall, ndcg_score))

        return precision, recall, ndcg_score

