import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from graphsaint.utils import *
import graphsaint.pytorch_version.layers as layers


class GraphSAINT(nn.Module):
    def __init__(self, num_classes, arch_gcn, train_params, feat_full, label_full, cpu_eval=False):
        """
        Build the multi-layer GNN architecture.

        Inputs:
            num_classes         int, number of classes a node can belong to
            arch_gcn            dict, config for each GNN layer
            train_params        dict, training hyperparameters (e.g., learning rate)
            feat_full           np array of shape N x f, where N is the total num of
                                nodes and f is the dimension for input node feature
            label_full          np array, for single-class classification, the shape
                                is N x 1 and for multi-class classification, the
                                shape is N x c (where c = num_classes)
            cpu_eval            bool, if True, will put the model on CPU.

        Outputs:
            None
        """
        super(GraphSAINT, self).__init__()
        self.use_cuda = (Globals.args_global.gpu >= 0)
        if cpu_eval:
            self.use_cuda = False
        if "attention" in arch_gcn:
            if "gated_attention" in arch_gcn:
                if arch_gcn['gated_attention']:
                    self.aggregator_cls = layers.GatedAttentionAggregator
                    self.mulhead = int(arch_gcn['attention'])
            else:
                self.aggregator_cls = layers.AttentionAggregator
                self.mulhead = int(arch_gcn['attention'])
        else:
            self.aggregator_cls = layers.HighOrderAggregator
            self.mulhead = 1
        self.num_layers = len(arch_gcn['arch'].split('-'))
        self.weight_decay = train_params['weight_decay']
        self.dropout = train_params['dropout']
        self.lr = train_params['lr']
        self.arch_gcn = arch_gcn
        self.sigmoid_loss = (arch_gcn['loss'] == 'sigmoid')
        self.feat_full = torch.from_numpy(feat_full.astype(np.float32))
        self.label_full = torch.from_numpy(label_full.astype(np.float32))
        if self.use_cuda:
            self.feat_full = self.feat_full.cuda()
            self.label_full = self.label_full.cuda()
        if not self.sigmoid_loss:
            self.label_full_cat = torch.from_numpy(label_full.argmax(axis=1).astype(np.int64))
            if self.use_cuda:
                self.label_full_cat = self.label_full_cat.cuda()
        self.num_classes = num_classes
        _dims, self.order_layer, self.act_layer, self.bias_layer, self.aggr_layer \
            = parse_layer_yml(arch_gcn, self.feat_full.shape[1])
        # get layer index for each conv layer, useful for jk net last layer aggregation
        self.set_idx_conv()
        self.set_dims(_dims)

        self.loss = 0
        self.opt_op = None

        # build the model below
        self.num_params = 0
        self.aggregators, num_param = self.get_aggregators()
        self.num_params += num_param
        self.conv_layers = nn.Sequential(*self.aggregators)
        if Globals.args_global.loss_action == 'mul':
            self.classifier = layers.HighOrderAggregator(self.dims_feat[-1], self.num_classes,
                                                         act='I', order=0, dropout=self.dropout, bias='bias')
        elif Globals.args_global.loss_action == 'cat':
            self.classifier = layers.HighOrderAggregator(2 * self.dims_feat[-1], self.num_classes,
                                                         act='I', order=0, dropout=self.dropout, bias='bias')
        self.num_params += self.classifier.num_param
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def set_dims(self, dims):
        """
        Set the feature dimension / weight dimension for each GNN or MLP layer.
        We will use the dimensions set here to initialize PyTorch layers.

        Inputs:
            dims        list, length of node feature for each hidden layer

        Outputs:
            None
        """
        self.dims_feat = [dims[0]] + [
            ((self.aggr_layer[l] == 'concat') * self.order_layer[l] + 1) * dims[l + 1]
            for l in range(len(dims) - 1)
        ]
        self.dims_weight = [(self.dims_feat[l], dims[l + 1]) for l in range(len(dims) - 1)]

    def set_idx_conv(self):
        """
        Set the index of GNN layers for the full neural net. For example, if
        the full NN is having 1-0-1-0 arch (1-hop graph conv, followed by 0-hop
        MLP, ...). Then the layer indices will be 0, 2.
        """
        idx_conv = np.where(np.array(self.order_layer) >= 1)[0]
        idx_conv = list(idx_conv[1:] - 1)
        idx_conv.append(len(self.order_layer) - 1)
        _o_arr = np.array(self.order_layer)[idx_conv]
        if np.prod(np.ediff1d(_o_arr)) == 0:
            self.idx_conv = idx_conv
        else:
            self.idx_conv = list(np.where(np.array(self.order_layer) == 1)[0])

    def forward(self, node_subgraph, adj_subgraph, adj_full_norm):
        edge_subgraph = self.get_edge_subgraph(node_subgraph, adj_subgraph, adj_full_norm)
        subgraph = node_subgraph if Globals.args_global.loss_type == 'node' else edge_subgraph
        feat_subg = self.feat_full[node_subgraph]
        label_subg = self.label_full[subgraph]
        label_subg_converted = label_subg if self.sigmoid_loss else label_subg  # self.label_full_cat[subgraph]
        _, emb_subg = self.conv_layers((adj_subgraph, feat_subg))
        emb_subg_norm = F.normalize(emb_subg, p=2, dim=1)
        # pred_subg = self.classifier((None, emb_subg_norm))[1]
        pred_subg = self.classifier((None, emb_subg_norm))[1] if Globals.args_global.loss_type == 'node' else \
            self.get_edge_pred_subg(emb_subg_norm, adj_subgraph, action=Globals.args_global.loss_action)
        return pred_subg, label_subg, label_subg_converted

    def _loss(self, preds, labels, norm_loss):
        """
        The predictor performs sigmoid (for multi-class) or softmax (for single-class)
        """
        if self.sigmoid_loss:
            norm_loss = norm_loss.unsqueeze(1)
            return torch.nn.BCEWithLogitsLoss(weight=norm_loss, reduction='sum')(preds, labels)
        else:
            _ls = torch.nn.CrossEntropyLoss(reduction='none')(preds, labels)
            return (norm_loss * _ls).sum()

    def get_aggregators(self):
        """
        Return a list of aggregator instances. to be used in self.build()
        """
        num_param = 0
        aggregators = []
        for l in range(self.num_layers):
            aggr = self.aggregator_cls(
                *self.dims_weight[l],
                dropout=self.dropout,
                act=self.act_layer[l],
                order=self.order_layer[l],
                aggr=self.aggr_layer[l],
                bias=self.bias_layer[l],
                mulhead=self.mulhead,
            )
            num_param += aggr.num_param
            aggregators.append(aggr)
        return aggregators, num_param

    def predict(self, preds):
        return nn.Sigmoid()(preds) if self.sigmoid_loss else F.softmax(preds, dim=1)[:, -1]

    def train_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph, minibatch):
        """
        Forward and backward propagation
        """
        self.train()
        self.optimizer.zero_grad()
        preds, labels, labels_converted = self(node_subgraph, adj_subgraph, minibatch.adj_full_norm)
        if Globals.args_global.loss_type == 'node':
            loss = self._loss(preds, labels_converted, norm_loss_subgraph)  # labels.squeeze()?
        elif Globals.args_global.loss_type == 'edge':
            loss = self.costum_loss(preds, labels_converted, norm_loss_subgraph)
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        return loss, self.predict(preds), labels

    def eval_step(self, node_subgraph, adj_subgraph, norm_loss_subgraph, minibatch):
        """
        Forward propagation only
        """
        self.eval()
        with torch.no_grad():
            preds, labels, labels_converted = self(node_subgraph, adj_subgraph, minibatch.adj_full_norm)

            if Globals.args_global.loss_type == 'node':
                loss = self._loss(preds, labels_converted, norm_loss_subgraph)  # labels.squeeze()?
            elif Globals.args_global.loss_type == 'edge':
                loss = self.costum_loss(preds, labels_converted, norm_loss_subgraph)
        return loss, self.predict(preds), labels

    def get_edge_subgraph(self, node_subgraph, adj_subgraph, adj_full_norm):
        edge_subgraph = np.array([])
        original_to_new = {val: i for i, val in enumerate(node_subgraph)}
        for i, (idx1, idx2) in enumerate(zip(adj_full_norm._indices()[0], adj_full_norm._indices()[1])):
            if int(idx1) in original_to_new.keys() and int(idx2) in original_to_new.keys() and \
                    adj_subgraph[original_to_new[int(idx1)]][original_to_new[int(idx2)]]:
                edge_subgraph = np.append(edge_subgraph, int(i))
        return edge_subgraph.astype(int)

    def get_edge_pred_subg(self, emb_subg_norm, adj_subgraph, action='mul'):
        edge_pred_subg = torch.FloatTensor([])
        if self.use_cuda:
            edge_pred_subg = edge_pred_subg.cuda()
        for i, (idx1, idx2) in enumerate(zip(adj_subgraph._indices()[0], adj_subgraph._indices()[1])):
            if action == 'mul':
                edge_pred_subg = torch.cat(
                    (edge_pred_subg, torch.matmul(emb_subg_norm[idx1], emb_subg_norm[idx2]).unsqueeze(0)))
                # Dim = [N]
            else:
                assert action == 'cat'
                edge_pred_subg = torch.cat(
                    (edge_pred_subg, torch.cat((emb_subg_norm[idx1], emb_subg_norm[idx2])).unsqueeze(0)))
                # Dim = [N, 256]

            # if int(idx1) in original_to_new.keys() and int(idx2) in original_to_new.keys() and \
            #    adj_subgraph[original_to_new[int(idx1)]][original_to_new[int(idx2)]]:
            #     edge_pred_subg = torch.cat(edge_pred_subg, )
        if action == 'cat':
            edge_pred_subg = self.classifier((None, edge_pred_subg))[1]
            # Dim = [N, 2]
        # else:
        #     edge_pred_subg = torch.mul(edge_pred_subg, -1)
        return edge_pred_subg

    # def get_edge_norm_loss(self, norm_loss):
    #     edge_pred_subg = torch.FloatTensor([])
    #     for i, (idx1, idx2) in enumerate(zip(adj_subgraph._indices()[0], adj_subgraph._indices()[1])):
    #         edge_pred_subg = torch.cat((edge_pred_subg, (pred_subg[idx1] * pred_subg[idx2]).unsqueeze(0)))
    #         # if int(idx1) in original_to_new.keys() and int(idx2) in original_to_new.keys() and \
    #         #    adj_subgraph[original_to_new[int(idx1)]][original_to_new[int(idx2)]]:
    #         #     edge_pred_subg = torch.cat(edge_pred_subg, )
    #     return edge_pred_subg

    def costum_loss(self, preds, labels, norm_loss):
        # norm_loss = norm_loss.unsqueeze(1)
        if Globals.args_global.loss_action == 'mul':
            zeros = torch.zeros(preds.size()[0], 1)
            preds = preds.unsqueeze(1)
            #
            # preds = torch.cat((preds, zeros), 1)
            preds = torch.cat((zeros, preds), 1)

            # Dim = [N, 2]
        labels = labels.float()
        return torch.nn.BCEWithLogitsLoss(reduction='mean')(preds, labels)
        # preds = preds.unsqueeze(1)
        # _ls = torch.nn.CrossEntropyLoss(reduction='none')(preds, labels)
        #
        # return _ls.sum() / len(_ls)
