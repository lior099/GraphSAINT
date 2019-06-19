from graphsaint.globals import *
import math
from graphsaint.inits import *
from graphsaint.utils import *
from graphsaint.graph_samplers import *
from graphsaint.norm_aggr import *
import tensorflow as tf
import scipy.sparse as sp
import scipy

import numpy as np
import time

import pdb




class Minibatch:
    """
    This minibatch iterator iterates over nodes for supervised learning.
    """

    def __init__(self, adj_full, adj_full_norm, adj_train, role, class_arr, placeholders, train_params, **kwargs):
        """
        role:       array of string (length |V|)
                    storing role of the node ('tr'/'va'/'te')
        class_arr: array of float (shape |V|xf)
                    storing initial feature vectors
        TODO:       adj_full_norm: normalized adj. Norm by # non-zeros per row

        norm_aggr_train:        list of dict
                                norm_aggr_train[u] gives a dict recording prob of being sampled for all u's neighbor
                                norm_aggr_train[u][v] gives the prob of being sampled for edge (u,v). i.e. P_{sampler}(u sampled|v sampled)
        """
        self.num_proc = 1
        self.node_train = np.array(role['tr'])
        self.node_val = np.array(role['va'])
        self.node_test = np.array(role['te'])

        self.class_arr = class_arr
        self.adj_full = adj_full
        self.adj_full_norm = adj_full_norm
        s1=int(adj_full_norm.shape[0]/8*1)
        s2=int(adj_full_norm.shape[0]/8*2)
        s3=int(adj_full_norm.shape[0]/8*3)
        s4=int(adj_full_norm.shape[0]/8*4)
        s5=int(adj_full_norm.shape[0]/8*5)
        s6=int(adj_full_norm.shape[0]/8*6)
        s7=int(adj_full_norm.shape[0]/8*7)
        self.adj_full_norm_0=adj_full_norm[:s1,:]
        self.adj_full_norm_1=adj_full_norm[s1:s2,:]
        self.adj_full_norm_2=adj_full_norm[s2:s3,:]
        self.adj_full_norm_3=adj_full_norm[s3:s4,:]
        self.adj_full_norm_4=adj_full_norm[s4:s5,:]
        self.adj_full_norm_5=adj_full_norm[s5:s6,:]
        self.adj_full_norm_6=adj_full_norm[s6:s7,:]
        self.adj_full_norm_7=adj_full_norm[s7:,:]
        self.adj_train = adj_train

        assert self.class_arr.shape[0] == self.adj_full.shape[0]

        # below: book-keeping for mini-batch
        self.placeholders = placeholders
        self.node_subgraph = None
        self.batch_num = -1

        self.method_sample = None
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_indices_orig = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        
        self.norm_loss_train = None
        self.norm_loss_test = np.zeros(self.adj_full.shape[0])
        _denom = len(self.node_train) + len(self.node_val) +  len(self.node_test)
        self.norm_loss_test[self.node_train] = 1/_denom
        self.norm_loss_test[self.node_val] = 1/_denom
        self.norm_loss_test[self.node_test] = 1/_denom
        self.norm_aggr_train = [dict()]     # list of dict. List index: start node index. dict key: end node idx
       
        self.sample_coverage = train_params['sample_coverage']
        self.deg_train = np.array(self.adj_train.sum(1)).flatten()


    def set_sampler(self,train_phases):
        self.subgraphs_remaining_indptr = list()
        self.subgraphs_remaining_indices = list()
        self.subgraphs_remaining_indices_orig = list()
        self.subgraphs_remaining_data = list()
        self.subgraphs_remaining_nodes = list()
        self.method_sample = train_phases['sampler']
        if self.method_sample == 'mrw':
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = mrw_sampling(self.adj_train,\
                self.node_train,self.size_subg_budget,train_phases['size_frontier'],int(train_phases['max_deg']))
        elif self.method_sample == 'rw':
            self.size_subg_budget = train_phases['num_root']*train_phases['depth']
            self.graph_sampler = rw_sampling(self.adj_train,\
                self.node_train,self.size_subg_budget,int(train_phases['num_root']),int(train_phases['depth']))
        elif self.method_sample == 'edge':
            self.size_subg_budget = train_phases['size_subg_edge']*2
            self.graph_sampler = edge_sampling(self.adj_train,self.node_train,train_phases['size_subg_edge'])
        elif self.method_sample == 'node':
            self.size_subg_budget = train_phases['size_subgraph']
            self.graph_sampler = node_sampling(self.adj_train,self.node_train,self.size_subg_budget)
        else:
            raise NotImplementedError

        def _init_norm_aggr_cnt(val_init):
            self.norm_aggr_train = list()
            for u in range(self.adj_train.shape[0]):
                self.norm_aggr_train.append(dict())
                _ind_start = self.adj_train.indptr[u]
                _ind_end = self.adj_train.indptr[u+1]
                for v in self.adj_train.indices[_ind_start:_ind_end]:
                    self.norm_aggr_train[u][v] = val_init

        self.norm_loss_train = np.zeros(self.adj_full.shape[0])
        tot_sampled_nodes = 0
        # 1. sample enough subg
        while True:
            self.par_graph_sample('train')
            tot_sampled_nodes = sum([len(n) for n in self.subgraphs_remaining_nodes])
            if tot_sampled_nodes > self.sample_coverage*self.node_train.size:
                break
        num_subg = len(self.subgraphs_remaining_nodes)
        avg_subg_size = tot_sampled_nodes/num_subg
        # 2. update _node_cnt --> to be used by norm_loss_train/norm_aggr_train
        _node_cnt = np.zeros(self.adj_full.shape[0])
        for i in range(num_subg):
            _node_cnt[self.subgraphs_remaining_nodes[i]] += 1
        # 3. norm_loss based on _node_cnt
        self.norm_loss_train[:] = _node_cnt[:]
        assert self.norm_loss_train[self.node_val].sum() + self.norm_loss_train[self.node_test].sum() == 0
        self.norm_loss_train[np.where(self.norm_loss_train==0)[0]] = 0.1
        self.norm_loss_train[self.node_val] = 0
        self.norm_loss_train[self.node_test] = 0
        self.norm_loss_train[self.node_train] = num_subg/self.norm_loss_train[self.node_train]/self.node_train.size
        # 4. norm_aggr based on _node_cnt and edge count
        _init_norm_aggr_cnt(0)
        self.norm_aggr_train_dok=sp.dok_matrix(self.adj_train.shape,dtype=scipy.float32)
        for i in range(num_subg):
            for ip in range(len(self.subgraphs_remaining_nodes[i])):
                _u = self.subgraphs_remaining_nodes[i][ip]
                for _v in self.subgraphs_remaining_indices_orig[i][self.subgraphs_remaining_indptr[i][ip]:self.subgraphs_remaining_indptr[i][ip+1]]:
                    self.norm_aggr_train[_u][_v] += 1
        self.norm_aggr_train_csr_data=[]
        for i_d,d in enumerate(self.norm_aggr_train):
            for k in sorted(d):
                v=d[k]
                self.norm_aggr_train_csr_data.append((_node_cnt[i_d])/(v+(v==0)*0.1)+(_node_cnt[i_d]==0)*0.1)
        self.norm_aggr_train_csr_data=np.array(self.norm_aggr_train_csr_data,dtype=np.float32)


    def par_graph_sample(self,phase):
        t0 = time.time()
        # _indices_orig: subgraph with indices in the original graph
        _indptr,_indices,_indices_orig,_data,_v,_edge_index= self.graph_sampler.par_sample(phase)
        t1 = time.time()
        print('sampling 200 subgraphs:   time = ',t1-t0)
        self.subgraphs_remaining_indptr.extend(_indptr)
        self.subgraphs_remaining_indices.extend(_indices)
        self.subgraphs_remaining_indices_orig.extend(_indices_orig)
        self.subgraphs_remaining_data.extend(_data)
        self.subgraphs_remaining_nodes.extend(_v)
        self.subgraphs_remaining_edge_index.extend(_edge_index)

    def feed_dict(self,dropout,mode='train'):
        """ DONE """
        if mode in ['val','test']:
            self.node_subgraph = np.arange(self.class_arr.shape[0])
            adj = sp.csr_matrix(([],[],np.zeros(2)), shape=(1,self.node_subgraph.shape[0]))
            adj_0 = self.adj_full_norm_0
            adj_1 = self.adj_full_norm_1
            adj_2 = self.adj_full_norm_2
            adj_3 = self.adj_full_norm_3
            adj_4 = self.adj_full_norm_4
            adj_5 = self.adj_full_norm_5
            adj_6 = self.adj_full_norm_6
            adj_7 = self.adj_full_norm_7
        else:
            assert mode == 'train'
            tt0=time.time()
            if len(self.subgraphs_remaining_nodes) == 0:
                self.par_graph_sample('train')

            self.node_subgraph = self.subgraphs_remaining_nodes.pop()
            self.size_subgraph = len(self.node_subgraph)
            adj = sp.csr_matrix((self.subgraphs_remaining_data.pop(),self.subgraphs_remaining_indices.pop(),\
                        self.subgraphs_remaining_indptr.pop()),shape=(self.node_subgraph.size,self.node_subgraph.size))
            adj_edge_index=self.subgraphs_remaining_edge_index.pop()
            #print("{} nodes, {} edges, {} degree".format(self.node_subgraph.size,adj.size,adj.size/self.node_subgraph.size))
            D = self.deg_train[self.node_subgraph]
            tt1 = time.time()
            assert len(self.node_subgraph) == adj.shape[0]
            norm_aggr(adj.data,adj_edge_index,self.norm_aggr_train_csr_data)
            # adj.data=self.norm_aggr_train_csr_data[adj_edge_index]

            tt2 = time.time()
            adj = sp.dia_matrix((1/D,0),shape=(adj.shape[0],adj.shape[1])).dot(adj)

            adj_0 = sp.csr_matrix(([],[],np.zeros(2)),shape=(1,self.node_subgraph.shape[0]))
            adj_1 = sp.csr_matrix(([],[],np.zeros(2)),shape=(1,self.node_subgraph.shape[0]))
            adj_2 = sp.csr_matrix(([],[],np.zeros(2)),shape=(1,self.node_subgraph.shape[0]))
            adj_3 = sp.csr_matrix(([],[],np.zeros(2)),shape=(1,self.node_subgraph.shape[0]))
            adj_4 = sp.csr_matrix(([],[],np.zeros(2)),shape=(1,self.node_subgraph.shape[0]))
            adj_5 = sp.csr_matrix(([],[],np.zeros(2)),shape=(1,self.node_subgraph.shape[0]))
            adj_6 = sp.csr_matrix(([],[],np.zeros(2)),shape=(1,self.node_subgraph.shape[0]))
            adj_7 = sp.csr_matrix(([],[],np.zeros(2)),shape=(1,self.node_subgraph.shape[0]))
            self.batch_num += 1
        feed_dict = dict()
        feed_dict.update({self.placeholders['node_subgraph']: self.node_subgraph})
        feed_dict.update({self.placeholders['labels']: self.class_arr[self.node_subgraph]})
        feed_dict.update({self.placeholders['dropout']: dropout})
        if mode in ['val','test']:
            feed_dict.update({self.placeholders['norm_loss']: self.norm_loss_test})
        else:
            feed_dict.update({self.placeholders['norm_loss']: self.norm_loss_train})
        
        _num_edges = len(adj.nonzero()[1])
        _num_vertices = len(self.node_subgraph)
        _indices_ph = np.column_stack(adj.nonzero())
        _shape_ph = adj.shape
        feed_dict.update({self.placeholders['adj_subgraph']: \
            tf.SparseTensorValue(_indices_ph,adj.data,_shape_ph)})
        feed_dict.update({self.placeholders['adj_subgraph_0']: \
            tf.SparseTensorValue(np.column_stack(adj_0.nonzero()),adj_0.data,adj_0.shape)})
        feed_dict.update({self.placeholders['adj_subgraph_1']: \
            tf.SparseTensorValue(np.column_stack(adj_1.nonzero()),adj_1.data,adj_1.shape)})
        feed_dict.update({self.placeholders['adj_subgraph_2']: \
            tf.SparseTensorValue(np.column_stack(adj_2.nonzero()),adj_2.data,adj_2.shape)})
        feed_dict.update({self.placeholders['adj_subgraph_3']: \
            tf.SparseTensorValue(np.column_stack(adj_3.nonzero()),adj_3.data,adj_3.shape)})
        feed_dict.update({self.placeholders['adj_subgraph_4']: \
            tf.SparseTensorValue(np.column_stack(adj_4.nonzero()),adj_4.data,adj_4.shape)})
        feed_dict.update({self.placeholders['adj_subgraph_5']: \
            tf.SparseTensorValue(np.column_stack(adj_5.nonzero()),adj_5.data,adj_5.shape)})
        feed_dict.update({self.placeholders['adj_subgraph_6']: \
            tf.SparseTensorValue(np.column_stack(adj_6.nonzero()),adj_6.data,adj_6.shape)})
        feed_dict.update({self.placeholders['adj_subgraph_7']: \
            tf.SparseTensorValue(np.column_stack(adj_7.nonzero()),adj_7.data,adj_7.shape)})
        tt3=time.time()
        # if mode in ['train']:
        #     print("t1:{:.3f} t2:{:.3f} t3:{:.3f}".format(tt0-tt1,tt2-tt1,tt3-tt2))
        if mode in ['val','test']:
            feed_dict[self.placeholders['is_train']]=False
        else:
            feed_dict[self.placeholders['is_train']]=True
        return feed_dict, self.class_arr[self.node_subgraph]


    def num_training_batches(self):
        """ DONE """
        return math.ceil(self.node_train.shape[0]/float(self.size_subg_budget))

    def shuffle(self):
        """ DONE """
        self.node_train = np.random.permutation(self.node_train)
        self.batch_num = -1

    def end(self):
        """ DONE """
        return (self.batch_num+1)*self.size_subg_budget >= self.node_train.shape[0]
