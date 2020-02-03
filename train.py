import os
import argparse
import time
import numpy as np
import math
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from DataLoad import MyDataset
from Model import MMGCN
from MMGraphSAGE import MMGraphSAGE
import pdb
import sys

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = -np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = math.floor(val_loss*10000)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            f.write(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), f'({args.l_r:.6f}_{args.weight_decay:.6f})_checkpoint.pt')
        self.val_loss_min = val_loss

# log
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



class Net:
    def __init__(self, args):
        ##########################################################################################################################################
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        ##########################################################################################################################################
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.learning_rate = args.l_r#l_r#
        self.weight_decay = args.weight_decay#weight_decay#
        self.batch_size = args.batch_size
        self.concat = args.concat
        self.num_workers = args.num_workers
        self.num_epoch = args.num_epoch
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.dim_latent = args.dim_latent
        self.aggr_mode = args.aggr_mode#aggr_mode#
        self.num_layer = args.num_layer
        self.has_id = args.has_id
        self.dim_v = 2048
        self.dim_a = 128
        self.dim_t = 100
        ##########################################################################################################################################
        print('Data loading ...')
        self.train_dataset = MyDataset('./Data/', self.num_user, self.num_item)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.edge_index = np.load('./Data/train.npy')
        self.user_index_5 = np.load('./Data/edge_adj_12.npy')
        self.val_dataset = np.load('./Data/val.npy')
        self.test_dataset = np.load('./Data/test.npy')
        self.val_test_dataset = np.load('./Data/val_test.npy')
        self.v_feat = np.load('./Data/FeatureVideo_normal.npy')
        self.a_feat = np.load('./Data/FeatureAudio_avg_normal.npy')
        self.t_feat = np.load('./Data/FeatureText_stl_normal.npy')
        print('Data has been loaded.')
        ##########################################################################################################################################
        if self.model_name == 'MMGCN':
            self.features = [self.v_feat, self.a_feat, self.t_feat]
            self.model = MMGCN(self.features, self.edge_index, self.user_index_5,self.batch_size, self.num_user, self.num_item, self.aggr_mode, self.concat, self.num_layer, self.has_id, self.dim_latent).to(self.device)

        elif self.model_name == 'VBPR':
            self.dim_feat = self.dim_v+self.dim_a+self.dim_t
            self.features = torch.tensor(np.concatenate((self.v_feat, self.a_feat, self.t_feat), axis=1), dtype=torch.float)
            self.model = VBPR_model(self.num_user, self.num_item, self.dim_latent, self.dim_feat, self.features).to(self.device)

        elif self.model_name == 'GraphSAGE':
            self.dim_feat = self.dim_v+self.dim_a+self.dim_t
            self.features = torch.tensor(np.concatenate((self.v_feat, self.a_feat, self.t_feat), axis=1), dtype=torch.float)
            self.model = MMGraphSAGE(self.features, self.edge_index, self.batch_size, self.num_user, self.num_item, self.dim_latent, self.dim_feat).cuda()

        elif self.model_name == 'NGCF':
            self.model = NGCF(self.v_feat, self.a_feat, self.t_feat, self.edge_index, self.batch_size, self.num_user, self.num_item, self.dim_latent).cuda()

        if args.PATH_weight_load and os.path.exists(args.PATH_weight_load):
            self.model.load_state_dict(torch.load(args.PATH_weight_load))
            print('module weights loaded....')
        ##########################################################################################################################################
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.learning_rate}], weight_decay=self.weight_decay)
        ##########################################################################################################################################

    def run(self):
        max_recall = 0.0
        max_pre = 0.0
        max_ndcg = 0.0
        step = 0

        print(args.l_r)
        print(args.weight_decay)
        with open(f'./train_log_({args.l_r:.6f}_{args.weight_decay:.6f}).txt', "w") as f:
            f.write(f'./train_log_({args.l_r:.6f}_{args.weight_decay:.6f}).txt')
            f.write("\n")
        early_stopping = EarlyStopping(patience=20, verbose=True)
        for epoch in range(self.num_epoch):
            self.model.train()
            print('Now, training start ...')
            pbar = tqdm(total=len(self.train_dataset))
            sum_loss = 0.0
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                self.loss = self.model.loss(data)
                self.loss.backward()
                self.optimizer.step()
                pbar.update(self.batch_size)
                sum_loss += self.loss
            print(sum_loss/self.batch_size)
            pbar.close()

            if epoch % 1 == 0:
                print('Validation start...')
                self.model.eval()
                with torch.no_grad():
                    precision, recall, ndcg_score = self.model.accuracy(self.val_dataset, topk=10)
                    print('---------------------------------Val: {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                        epoch, 10, precision, recall, ndcg_score))
                    with open(f'./train_log_({args.l_r:.6f}_{args.weight_decay:.6f}).txt', "w") as f:
                         f.write('---------------------------------Val: {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                        epoch, 10, precision, recall, ndcg_score))  # 将字符串写入文件中
                         f.write("\n")
            early_stopping(precision, self.model)

            if early_stopping.early_stop:
                f.write("Early stopping")
                f.write("\n")
                break



            if epoch % 5 == 0:
                self.model.eval()
                with torch.no_grad():
                    precision, recall, ndcg_score = self.model.accuracy(self.test_dataset, topk=10)
                    print('---------------------------------Test: {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                        epoch, 10, precision, recall, ndcg_score))

        return max_recall, max_pre, max_ndcg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--model_name', default='MMGCN', help='Model name.')
    parser.add_argument('--data_path', default='amazon-book', help='Dataset path')
    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default=None, help='Writing weight filename.')
    parser.add_argument('--l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--dim_latent', type=int, default=64, help='Latent dimension.')
    parser.add_argument('--num_epoch', type=int, default=200, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=40, help='Workers number.')
    parser.add_argument('--num_user', type=int, default=55485, help='User number.')
    parser.add_argument('--num_item', type=int, default=5986, help='Item number.')
    parser.add_argument('--aggr_mode', default='mean', help='Aggregation mode.')
    parser.add_argument('--concat', type=bool, default=True, help='Concatenation')
    parser.add_argument('--num_layer', type=int, default=2, help='Layer number.')
    parser.add_argument('--has_id', type=bool, default=True, help='Has id_embedding')


    args = parser.parse_args()
    # sys.stdout = Logger(f'./train_log_({args.l_r:.6f}_{args.weight_decay:.6f}).txt')

    egcn = Net(args)
    egcn.run()    


