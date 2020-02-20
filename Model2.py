import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from BaseModel import BaseModel
from BaseModel import BaseModel_gcn
import pdb
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class GCN(torch.nn.Module):
    def __init__(self, features, edge_index, batch_size, num_user, num_item, dim_id, aggr_mode, concate, num_layer, has_id, dim_latent=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.features = features
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id

        if self.dim_latent:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).to(device)
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
            # nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_latent, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight) 

        else:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).to(device)
            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_feat, self.dim_id)     
            nn.init.xavier_normal_(self.g_layer1.weight)              
          
        self.conv_embed_2 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        # nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)    

        self.conv_embed_3 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        # nn.init.xavier_normal_(self.conv_embed_3.weight)
        self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer3.weight)
        self.g_layer3 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)    
    def forward(self,id_embedding):
        temp_features = self.MLP(self.features) if self.dim_latent else self.features
        x = torch.cat((self.preference, temp_features),dim=0)
        x = F.normalize(x).to(device)

        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))#equation 1
        # x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer1(x))#equation 5
        # x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer1(h)+x_hat)
        #
        # if self.num_layer > 1:
        #     h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))#equation 1
        #     x_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer2(x))#equation 5
        #     x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer2(h)+x_hat)
        # if self.num_layer > 2:
        #     h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))#equation 1
        #     x_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer3(x))#equation 5
        #     x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer3(h)+x_hat)
        return h
class GCN_1(torch.nn.Module):
    def __init__(self, features, batch_size, num_user, num_item, dim_id, aggr_mode, concate, num_layer, has_id, dropout,dim_latent=None):
        super(GCN_1, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.features = features
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout

        if self.dim_latent:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).to(device))
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_latent,self.dropout,aggr=self.aggr_mode)
            # nn.init.xavier_normal_(self.conv_embed_1.weight)
            # self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            # nn.init.xavier_normal_(self.linear_layer1.weight)
            # self.g_layer1 = nn.Linear(self.dim_latent+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_latent, self.dim_id)
            # nn.init.xavier_normal_(self.g_layer1.weight)

        else:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).to(device))
            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)
            # nn.init.xavier_normal_(self.conv_embed_1.weight)
            # self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            # nn.init.xavier_normal_(self.linear_layer1.weight)
            # self.g_layer1 = nn.Linear(self.dim_feat+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_feat, self.dim_id)
            # nn.init.xavier_normal_(self.g_layer1.weight)
          
        # self.conv_embed_2 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        # # nn.init.xavier_normal_(self.conv_embed_2.weight)
        # self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        # nn.init.xavier_normal_(self.linear_layer2.weight)
        # self.g_layer2 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)
        #
        # self.conv_embed_3 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        # # nn.init.xavier_normal_(self.conv_embed_3.weight)
        # self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
        # nn.init.xavier_normal_(self.linear_layer3.weight)
        # self.g_layer3 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)

    def forward(self,edge_index):
        temp_features = self.MLP(self.features) if self.dim_latent else self.features
        x = torch.cat((self.preference, temp_features),dim=0)
        x = F.normalize(x).to(device)
        h = self.conv_embed_1(x,edge_index)#equation 1
        # h = self.conv_embed_1(h, self.edge_index)
        x_hat = h+x
        # x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer1(x))#equation 5
        # x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer1(h)+x_hat)
        #
        # if self.num_layer > 1:
        #     h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))#equation 1
        #     x_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer2(x))#equation 5
        #     x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer2(h)+x_hat)
        # if self.num_layer > 2:
        #     h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))#equation 1
        #     x_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer3(x))#equation 5
        #     x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer3(h)+x_hat)
        return x_hat,self.preference

class User_Graph(torch.nn.Module):
    def __init__(self,edge_index,num_user,aggr_mode):
        super(User_Graph,self).__init__()
        self.edge_index = edge_index
        self.num_user = num_user
        self.aggr_mode = aggr_mode
        self.base_gcn = BaseModel_gcn(64,64,self.aggr_mode)
    def forward(self,features):
        u_pre = self.base_gcn(features,self.edge_index)
        return u_pre

class MMGCN(torch.nn.Module):
    def __init__(self, features, edge_index,user_index_5, batch_size, num_user, num_item, aggr_mode, concate, num_layer, has_id, dim_x,reg_weight,dropout,pos_row, pos_col,user_item_dict):
        super(MMGCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.reg_weight = reg_weight
        self.user_item_dict = user_item_dict
        self.pos_row = torch.LongTensor(pos_row)
        self.pos_col = torch.LongTensor(pos_col)-num_user
        self.dropout = dropout
        self.v_rep = None
        self.a_rep = None
        self.t_rep = None
        self.v_preference = None
        self.a_preference = None
        self.t_preference = None

        self.edge_index = torch.tensor(edge_index).t().contiguous().to(device)
        # self.edge_index, _ = dropout_adj(edge_index, edge_attr=None, p=self.dropout)
        # self.edge_index = torch.cat((self.edge_index, self.edge_index[[1,0]]), dim=1)

        self.user_index_5 = torch.tensor(user_index_5).contiguous().to(device)
        v_feat, a_feat, t_feat = features
        self.v_feat = torch.tensor(v_feat, dtype=torch.float).to(device)
        self.a_feat = torch.tensor(a_feat, dtype=torch.float).to(device)
        self.t_feat = torch.tensor(t_feat, dtype=torch.float).to(device)

        self.v_gcn = GCN_1(self.v_feat, batch_size, num_user, num_item, dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id,dropout = self.dropout, dim_latent=64)#256)
        self.a_gcn = GCN_1(self.a_feat, batch_size, num_user, num_item, dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id,dropout = self.dropout, dim_latent=64)
        self.t_gcn = GCN_1(self.t_feat, batch_size, num_user, num_item, dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id,dropout = self.dropout, dim_latent=64)
        self.user_graph = User_Graph(self.user_index_5,num_user,'add')

        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x), requires_grad=True)).to(device)
        self.result_embed = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x)))).to(device)


    def forward(self, user_nodes, pos_item_nodes, neg_item_nodes):
        self.edge_index, _ = dropout_adj(self.edge_index, edge_attr=None, p=self.dropout)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        self.v_rep,self.v_preference = self.v_gcn(self.edge_index)
        self.a_rep,self.a_preference = self.a_gcn(self.edge_index)
        self.t_rep,self.t_preference = self.t_gcn(self.edge_index)
        # representation = (self.v_rep+self.a_rep+self.t_rep)/3
        representation = self.v_rep+self.a_rep+self.t_rep
        item_rep = representation[self.num_user:]
        user_rep = representation[:self.num_user]
        h_u1 = self.user_graph(user_rep)
        h_u2 = self.user_graph(h_u1)
        user_rep = user_rep+h_u1+h_u2
        self.result_embed = torch.cat((user_rep,item_rep),dim=0)
        # self.result_embed = representation
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        pos_scores = torch.sum(user_tensor*pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor*neg_item_tensor, dim=1)
        return pos_scores, neg_scores


    def loss(self, data):
        user, pos_items, neg_items = data
        pos_scores, neg_scores = self.forward(user.to(device), pos_items.to(device), neg_items.to(device))
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        # reg_embedding_loss = (self.result_embed[user] ** 2).mean()
        reg_embedding_loss_v = (self.v_preference[user.to(device)] ** 2).mean()
        reg_embedding_loss_a = (self.a_preference[user.to(device)] ** 2).mean()
        reg_embedding_loss_t = (self.t_preference[user.to(device)] ** 2).mean()
        reg_loss = self.reg_weight * (reg_embedding_loss_v+reg_embedding_loss_a+reg_embedding_loss_t)
        return loss_value+reg_loss,reg_loss

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