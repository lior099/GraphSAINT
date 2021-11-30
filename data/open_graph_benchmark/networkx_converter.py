import json
import os
import random

import networkx as nx
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp


SEP = '/'


def convert(graph, dir, y=None, future_graph=None, feats=None, mode='train_test', test_seed=0):
    if not os.path.exists(dir):
        os.makedirs(dir)
    size = len(graph.nodes)
    if mode == 'train_test':
        train_size = 0.6
        val_size = 0.2
    elif mode == 'train':
        train_size = 0.8
        val_size = 0.2
    elif mode == 'test':
        train_size = 0.00
        val_size = 0.00
    else:
        raise Exception("mode has to be train_test/train/test")

    # feats.npy
    # feats = get_feats(graph)
    np.save(SEP.join([dir, 'feats.npy']), feats)

    # role.json
    role = dict()
    indices = list(range(size))
    random.seed(test_seed)
    random.shuffle(indices)
    tr_idx, va_idx = int(train_size * size), int((train_size + val_size) * size)
    train_val_indices = indices[:va_idx]
    random.seed()
    random.shuffle(train_val_indices)
    indices[:va_idx] = train_val_indices
    role['tr'] = indices[:tr_idx]
    role['va'] = indices[tr_idx:va_idx]
    role['te'] = indices[va_idx:]
    with open(SEP.join([dir, 'role.json']), 'w') as f:
        json.dump(role, f)

    # class_map.json
    class_map = dict()
    y = get_y(graph, future_graph)
    for i in range(size):
        class_map[str(i)] = int(y[i])
    with open(SEP.join([dir, 'class_map.json']), 'w') as f:
        json.dump(class_map, f)

    # adj_*.npz
    train_idx_set = set(role['tr'])
    test_idx_set = set(role['te'])
    edge_index = list(zip(*graph.edges()))
    row_full = np.array(edge_index[0])
    col_full = np.array(edge_index[1])
    row_train = []
    col_train = []
    row_val = []
    col_val = []
    for i in tqdm(range(row_full.shape[0])):
        if row_full[i] in train_idx_set and col_full[i] in train_idx_set:
            row_train.append(row_full[i])
            col_train.append(col_full[i])
            row_val.append(row_full[i])
            col_val.append(col_full[i])
        elif not (row_full[i] in test_idx_set or col_full[i] in test_idx_set):
            row_val.append(row_full[i])
            col_val.append(col_full[i])
    row_train = np.array(row_train)
    col_train = np.array(col_train)
    row_val = np.array(row_val)
    col_val = np.array(col_val)
    dtype = np.bool

    adj_full = sp.coo_matrix(
        (
            np.ones(row_full.shape[0], dtype=dtype),
            (row_full, col_full),
        ),
        shape=(size, size)
    ).tocsr()

    adj_train = sp.coo_matrix(
        (
            np.ones(row_train.shape[0], dtype=dtype),
            (row_train, col_train),
        ),
        shape=(size, size)
    ).tocsr()

    adj_val = sp.coo_matrix(
        (
            np.ones(row_val.shape[0], dtype=dtype),
            (row_val, col_val),
        ),
        shape=(size, size)
    ).tocsr()

    # import pdb; pdb.set_trace()
    print('adj_full  num edges:', adj_full.nnz)
    print('adj_val   num edges:', adj_val.nnz)
    print('adj_train num edges:', adj_train.nnz)
    sp.save_npz(SEP.join([dir, 'adj_full.npz']), adj_full)
    sp.save_npz(SEP.join([dir, 'adj_train.npz']), adj_train)
    # adj_val not used in GraphSAINT
    sp.save_npz(SEP.join([dir, 'adj_val.npz']), adj_val)

def random_graph(size, p):
    # edges = [[i, j, 1] for i in range(size) for j in range(size) if i != j and random.random() <= p]
    # graph = nx.DiGraph()
    # graph.add_weighted_edges_from(edges)
    graph = nx.fast_gnp_random_graph(size, p, directed=True)
    return graph

def get_feats(graph):
    nodes_len = len(graph.nodes)
    feats_len = 5
    p = 0.2
    feats = [[0 if random.random() <= p else 0 for i in range(feats_len)] for j in range(nodes_len)]
    feats = np.array(feats)
    return feats

def get_y(graph1, graph2):
    # if len(graph1.nodes) != len(graph2.nodes):
    #     raise Exception("bad graphs")
    nodes_len = len(graph1.nodes)
    nodes_as_edges = [graph1.nodes[i] for i in range(nodes_len)]
    y = [1 if node['edge'] in graph2.nodes else 0 for node in nodes_as_edges]
    print(round(sum(y)/len(y), 3),"of the data has positive label (how many links are still connected in next snapshot)")
    # p = 0.2
    # y = [1 if random.random() <= p else 0 for i in range(nodes_len)]
    return y

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    size = 6000
    rank = 10
    p = rank / (size - 1)
    graph1 = random_graph(size, p)
    graph2 = random_graph(size, p)
    y = get_y(graph1, graph2)
    # convert(graph1, y, './data/test/')
