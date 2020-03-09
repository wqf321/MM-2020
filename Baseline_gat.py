import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from torch.nn import Parameter
from torch_geometric.nn import GATConv
import pdb


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class GAT(torch.nn.Module):
    def __init__(self,edge_index,dim_id,num_user,num_item,weight_decay,dropout ,user_item_dict):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dim_id,dim_id,heads=3,concat=False)
        # self.conv2 = GATConv(dim_id,dim_id,heads=3,concat=False)
        self.dropout =dropout
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.reg_weight = weight_decay
        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_id), requires_grad=True)).cuda()
        self.result_embed = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_id)))).cuda()
        self.edge_index = torch.tensor(edge_index).t().contiguous().cuda()

    def forward(self,user_nodes,pos_item_nodes,neg_item_nodes):
        edge_index, _ = dropout_adj(self.edge_index, edge_attr=None, p=self.dropout)
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)

        x = F.leaky_relu(self.conv1(self.id_embedding,edge_index))
        # x = F.leaky_relu(self.conv2(x,edge_index))

        self.result_embed = x
        # self.result_embed = representation
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)

        return pos_scores, neg_scores

    def loss(self, data):
        user, pos_items, neg_items = data
        pos_scores, neg_scores = self.forward(user.cuda(), pos_items.cuda(), neg_items.cuda())
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        # reg_embedding_loss = (self.result_embed[user] ** 2).mean()
        # reg_MLP = (self.MLP.weight ** 2).mean()
        reg_embedding_u = (self.id_embedding[user.cuda()] ** 2).mean()
        reg_embedding_pos = (self.id_embedding[pos_items.cuda()] ** 2).mean()
        reg_embedding_neg = (self.id_embedding[neg_items.cuda()] ** 2).mean()
        reg_loss = self.reg_weight * (reg_embedding_u+reg_embedding_neg+reg_embedding_pos)
        return loss_value + reg_loss, reg_loss

    def accuracy(self, step=2000, topk=10):
        user_tensor = self.result_embed[:self.num_user]
        item_tensor = self.result_embed[self.num_user:]

        start_index = 0
        end_index = self.num_user if step == None else step

        all_index_of_rank_list = torch.LongTensor([])
        while end_index <= self.num_user and start_index < end_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

            _, index_of_rank_list = torch.topk(score_matrix, topk)
            all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu() + self.num_user),
                                               dim=0)
            start_index = end_index

            if end_index + step < self.num_user:
                end_index += step
            else:
                end_index = self.num_user

        length = self.num_user
        precision = recall = ndcg = 0.0

        for row, col in self.user_item_dict.items():
            # col = np.array(list(col))-self.num_user
            user = row
            pos_items = set(col)
            # print(pos_items)
            num_pos = len(pos_items)
            if num_pos ==0:
                length = length-1
                continue
            items_list = all_index_of_rank_list[user].tolist()

            items = set(items_list)

            num_hit = len(pos_items.intersection(items))

            precision += float(num_hit / topk)
            recall += float(num_hit / num_pos)

            ndcg_score = 0.0
            max_ndcg_score = 0.0

            for i in range(min(num_hit, topk)):
                max_ndcg_score += 1 / math.log2(i + 2)
            if max_ndcg_score == 0:
                continue

            for i, temp_item in enumerate(items_list):
                if temp_item in pos_items:
                    ndcg_score += 1 / math.log2(i + 2)

            ndcg += ndcg_score / max_ndcg_score

        return precision / length, recall / length, ndcg / length

    def full_accuracy(self, val_data, step=2000, topk=10):
        user_tensor = self.result_embed[:self.num_user]
        item_tensor = self.result_embed[self.num_user:]

        start_index = 0
        end_index = self.num_user if step == None else step

        all_index_of_rank_list = torch.LongTensor([])
        while end_index <= self.num_user and start_index < end_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

            for row, col in self.user_item_dict.items():
                if row >= start_index and row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col)) - self.num_user
                    score_matrix[row][col] = 1e-5

            _, index_of_rank_list = torch.topk(score_matrix, topk)
            all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu() + self.num_user),
                                               dim=0)
            start_index = end_index

            if end_index + step < self.num_user:
                end_index += step
            else:
                end_index = self.num_user

        length = len(val_data)
        precision = recall = ndcg = 0.0

        for data in val_data:
            user = data[0]
            pos_items = set(data[1:])
            # print(pos_items)
            num_pos = len(pos_items)
            if num_pos ==0:
                length = length-1
                continue
            items_list = all_index_of_rank_list[user].tolist()

            items = set(items_list)

            num_hit = len(pos_items.intersection(items))

            precision += float(num_hit / topk)
            recall += float(num_hit / num_pos)

            ndcg_score = 0.0
            max_ndcg_score = 0.0

            for i in range(min(num_hit, topk)):
                max_ndcg_score += 1 / math.log2(i + 2)
            if max_ndcg_score == 0:
                continue

            for i, temp_item in enumerate(items_list):
                if temp_item in pos_items:
                    ndcg_score += 1 / math.log2(i + 2)

            ndcg += ndcg_score / max_ndcg_score

        return precision / length, recall / length, ndcg / length

