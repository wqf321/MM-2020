import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pdb
def gen_dense_user_graph(all_edge, num_inter):
    edge_dict = defaultdict(set)

    user_set = set()
    for edge in all_edge:
        user, item = edge
        edge_dict[user].add(item)
        user_set.add(user)

    user_list = list(user_set)
    user_list.sort()
    min_user = user_list[0]
    num_user = user_list[-1]-user_list[0]+1
    print(num_user)
    print(min_user)
    edge_set = set()
    edge_list_i = []
    edge_list_j = []

    edge_adj = np.zeros((num_user, num_user), dtype=np.int)#np.eyes((num_item, num_item), dtype=np.int)#
    key_list = list(edge_dict.keys())
    key_list.sort()
    bar = tqdm(total=len(key_list))
    node_set = set()
    for head in range(len(key_list)):
        bar.update(1)
        for rear in range(head, len(key_list)):
            head_key = key_list[head]
            rear_key = key_list[rear]
            # print(head_key, rear_key)
            item_head = edge_dict[head_key]
            item_rear = edge_dict[rear_key]
            # print(len(user_head.intersection(user_rear)))
            if len(item_head.intersection(item_rear)) > num_inter:
                # edge_list_i.append(head_key-min_item)
                # edge_list_j.append(rear_key-min_item)
                # edge_list_i.append([head_key-min_item, rear_key-min_item])
                # if head_key != rear_key:
                #         edge_list_j.append(head_key-min_item)
                #         edge_list_i.append(rear_key-min_item)
                        # edge_list.append([rear_key-min_item, head_key-min_item])
                edge_set.add((head_key, rear_key))
                node_set.add(head_key)
                node_set.add(rear_key)
                edge_adj[head_key-min_user, rear_key-min_user] = 1
                edge_adj[rear_key-min_user, head_key-min_user] = 1
    # edge_list = [edge_list_i, edge_list_j]
    bar.close()

    return edge_adj, edge_set, node_set
def gen_dense_item_graph(all_edge, num_inter):
    edge_dict = defaultdict(set)

    item_set = set()
    for edge in all_edge:
        user, item = edge
        edge_dict[item].add(user)
        item_set.add(item)

    item_list = list(item_set)
    item_list.sort()
    min_item = item_list[0]
    num_item = item_list[-1]-item_list[0]+1
    print(num_item)
    print(min_item)
    edge_set = set()
    edge_list_i = []
    edge_list_j = []

    edge_adj = np.zeros((num_item, num_item), dtype=np.int)#np.eyes((num_item, num_item), dtype=np.int)#
    key_list = list(edge_dict.keys())
    key_list.sort()
    bar = tqdm(total=len(key_list))
    node_set = set()
    for head in range(len(key_list)):
        bar.update(1)
        for rear in range(head, len(key_list)):
            head_key = key_list[head]
            rear_key = key_list[rear]
            # print(head_key, rear_key)
            user_head = edge_dict[head_key]
            user_rear = edge_dict[rear_key]
            # print(len(user_head.intersection(user_rear)))
            if len(user_head.intersection(user_rear)) > num_inter:
                # edge_list_i.append(head_key-min_item)
                # edge_list_j.append(rear_key-min_item)
                # edge_list_i.append([head_key-min_item, rear_key-min_item])
                # if head_key != rear_key:
                #         edge_list_j.append(head_key-min_item)
                #         edge_list_i.append(rear_key-min_item)       
                        # edge_list.append([rear_key-min_item, head_key-min_item])
                edge_set.add((head_key, rear_key))
                node_set.add(head_key)
                node_set.add(rear_key)
                edge_adj[head_key-min_item, rear_key-min_item] = 1
                edge_adj[rear_key-min_item, head_key-min_item] = 1
    # edge_list = [edge_list_i, edge_list_j]
    bar.close()

    return edge_adj, edge_set, node_set
def gen_user_graph(all_edge):
    edge_dict = defaultdict(set)

    user_set = set()
    for edge in all_edge:
    	user, item = edge
    	edge_dict[user].add(item)
    	user_set.add(user)

    user_list = list(user_set)
    user_list.sort()
    min_user = user_list[0]
    num_user = user_list[-1]-user_list[0]+1
    print(num_user)
    print(min_user)
    edge_set = set()
    edge_list_i = []
    edge_list_j = []

    # edge_adj = np.zeros((num_item, num_item), dtype=np.int)#np.eyes((num_item, num_item), dtype=np.int)#
    key_list = list(edge_dict.keys())
    key_list.sort()
    bar = tqdm(total=len(key_list))
    node_set = set()
    for head in range(len(key_list)):
        bar.update(1)
        for rear in range(head, len(key_list)):
            head_key = key_list[head]
            rear_key = key_list[rear]
            # print(head_key, rear_key)
            item_head = edge_dict[head_key]
            item_rear = edge_dict[rear_key]
            # print(len(user_head.intersection(user_rear)))
            if len(item_head.intersection(item_rear)) > 12:
                edge_list_i.append(head_key-min_user)
                edge_list_j.append(rear_key-min_user)
                # edge_list_i.append([head_key-min_item, rear_key-min_item])
                if head_key != rear_key:
                        edge_list_j.append(head_key-min_user)
                        edge_list_i.append(rear_key-min_user)
                        # edge_list.append([rear_key-min_item, head_key-min_item])
                edge_set.add((head_key, rear_key))
                node_set.add(head_key)
                node_set.add(rear_key)
                # edge_adj[head_key-min_item, rear_key-min_item] = 1
                # edge_adj[rear_key-min_item, head_key-min_item] = 1
    edge_list = [edge_list_i, edge_list_j]
    bar.close()

    return edge_list, edge_set, node_set
def gen_item_graph(all_edge):
    edge_dict = defaultdict(set)

    item_set = set()
    for edge in all_edge:
    	user, item = edge
    	edge_dict[item].add(user)
    	item_set.add(item)

    item_list = list(item_set)
    item_list.sort()
    min_item = item_list[0]
    num_item = item_list[-1]-item_list[0]+1
    print(num_item)
    print(min_item)
    edge_set = set()
    edge_list_i = []
    edge_list_j = []

    # edge_adj = np.zeros((num_item, num_item), dtype=np.int)#np.eyes((num_item, num_item), dtype=np.int)#
    key_list = list(edge_dict.keys())
    key_list.sort()
    bar = tqdm(total=len(key_list))
    node_set = set()
    for head in range(len(key_list)):
        bar.update(1)
        for rear in range(head, len(key_list)):
            head_key = key_list[head]
            rear_key = key_list[rear]
            # print(head_key, rear_key)
            user_head = edge_dict[head_key]
            user_rear = edge_dict[rear_key]
            # print(len(user_head.intersection(user_rear)))
            if len(user_head.intersection(user_rear)) > 5:
                edge_list_i.append(head_key-min_item)
                edge_list_j.append(rear_key-min_item)
                # edge_list_i.append([head_key-min_item, rear_key-min_item])
                if head_key != rear_key:
                        edge_list_j.append(head_key-min_item)
                        edge_list_i.append(rear_key-min_item)       
                        # edge_list.append([rear_key-min_item, head_key-min_item])
                edge_set.add((head_key, rear_key))
                node_set.add(head_key)
                node_set.add(rear_key)
                # edge_adj[head_key-min_item, rear_key-min_item] = 1
                # edge_adj[rear_key-min_item, head_key-min_item] = 1
    edge_list = [edge_list_i, edge_list_j]
    bar.close()

    return edge_list, edge_set, node_set

# def data_load(dir_str):
#     train_edge = np.load(dir_str+'/train.npy', allow_pickle=True)
#     val_edge = np.load(dir_str+'/val.npy', allow_pickle=True)#.item()
#     test_edge = np.load(dir_str+'/test.npy', allow_pickle=True)#.item()
#     item_adj = np.load(dir_str+'edge_adj.npy', allow_pickle=True)

#     user_set = set()
#     item_set = set()
#     for edge in train_edge:
#     	user, item = edge
#     	user_set.add(user)
#     	item_set.add(item)

#     for edge in val_edge:
#     	user = edge[0]
#     	items = edge[1:]
#     	user_set.add(user)
#     	item_set = item_set.union(set(items))

#     for edge in test_edge:
#     	user = edge[0]
#     	items = edge[1:]
#     	user_set.add(user)
#     	item_set = item_set.union(set(items))

#     user_list = list(user_set)
#     item_list = list(item_set)
#     user_list.sort()
#     item_list.sort()

#     print(len(user_list), len(item_list))
#     print(user_list[0], user_list[-1])
#     print(item_list[0], item_list[-1])

def gen_user_dict(dir_str):
    user_item_dict = defaultdict(set)
    train_edge = np.load(dir_str+'/train.npy')
    for edge in train_edge:
    	user, item = edge
    	user_item_dict[user].add(item)
    np.save(dir_str+'/user_item_dict.npy', np.array(user_item_dict))

def full_data_gen(dir_str):
    val_edge = np.load(dir_str+'/val.npy', allow_pickle=True)
    test_edge = np.load(dir_str+'/test.npy', allow_pickle=True)
    val_full_list = list()
    test_full_list = list()

    for edge in val_edge:
    	if len(edge) < 1002:
    		continue
    	user = edge[0]
    	items = edge[1001:]
    	val_full_list.append([user]+items)

    for edge in test_edge:
    	if len(edge) < 1002:
    		continue
    	user = edge[0]
    	items = edge[1001:]
    	test_full_list.append([user]+items)
    np.save(dir_str+'/val_full.npy', np.array(val_full_list))
    np.save(dir_str+'/test_full.npy', np.array(test_full_list))

if __name__ == 	'__main__':    
    train_data = np.load('./Data/train.npy')

    # gen_item_graph(train_data)
    # edge_adj, edge_set, node_set = gen_dense_user_graph(train_data, 3)
    edge_adj, edge_set, node_set = gen_user_graph(train_data)
    np.save('./Data/edge_adj_12.npy', edge_adj)

    # np.save('./Data/Kwai/edge_list.npy', edge_adj)
    print(len(edge_set))
    print(len(node_set))
    # data_load('./Data/movielens/')
    # gen_user_dict('./Data/Kwai')    
    # full_data_gen('./Data/Kwai')