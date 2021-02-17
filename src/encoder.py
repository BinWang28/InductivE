
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from torch.nn import init
import numpy as np
import dgl.function as fn


# *****************************************************************
# Model: RWGCN (edge type enabled)
class RWGCN_Layer(nn.Module):
    def __init__(self, in_feat, out_feat, num_edge_types, activation=None, self_loop=True, dropout=0.2, residual=False):
        super(RWGCN_Layer, self).__init__()

        self.num_edge_types = num_edge_types
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.self_loop = self_loop
        self.bias = True
        self.same_transform_matrix = False

        if residual:
            if self.in_feat != self.out_feat:
                self.res_fc = nn.Linear(
                    self.in_feat, self.out_feat, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        # weight for self-loop
        self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
        if self.bias:
            self.loop_bias_weight = nn.Parameter(torch.Tensor(out_feat))
            init.zeros_(self.loop_bias_weight)
            #stdv = 1. / np.sqrt(out_feat)
            #self.loop_bias_weight.data.uniform_(-stdv, stdv)

        # weight for other relation
        if self.same_transform_matrix:
            self.weight = self.loop_weight
        else:
            self.weight = nn.Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        # add bias
        if self.bias:
            if self.same_transform_matrix:
                self.bias_weight = self.loop_bias_weight
            else:
                self.bias_weight = nn.Parameter(torch.Tensor(out_feat))
                init.zeros_(self.bias_weight)
                #stdv = 1. / np.sqrt(out_feat)
                #self.bias_weight.data.uniform_(-stdv, stdv)
        else:
            self.bias_weight = None

        self.weight_rel = nn.Parameter(torch.FloatTensor(self.num_edge_types, 1))
        
        if self.self_loop:
            self.gate_self_loop = nn.Parameter(torch.FloatTensor(2, 1))
            self.gating_attention = nn.Parameter(torch.FloatTensor(self.out_feat*2, 1))

        self.bn = torch.nn.BatchNorm1d(self.out_feat)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.init()

    def init(self):

        init.xavier_uniform_(self.loop_weight)
        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.weight_rel)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight)
        
        init.xavier_uniform_(self.gate_self_loop)
        init.xavier_uniform_(self.gating_attention)



    def regularization_loss(self):
        
        return 0
        
        total_reg = 0
        # self loop
        total_reg += torch.mean(self.loop_weight.pow(2))
        # residual matrix
        if isinstance(self.res_fc, nn.Linear):
            total_reg += torch.mean(self.res_fc.weight.pow(2))
        # w2
        total_reg += torch.mean(self.loop_weight.pow(2))

        return total_reg


    def msg_func(self, edges):
        ''' compute messages only from source node features
        '''
        edge_types = edges.data['type'].squeeze()
        alpha = self.weight_rel[edge_types]
        node = torch.mm(edges.src['h'], self.weight)
        msg = alpha.expand_as(node) * node
        return {'msg': msg, 'alpha': alpha}

    def attn_reduce(self, nodes):
        ''' mean of neighboring edges
        '''
        attn_sum = torch.mean(nodes.mailbox['msg'],dim=1)
        #norm_sum = torch.sum(nodes.mailbox['alpha'],dim=1)
        #return {'h': attn_sum, 'norm_sum': norm_sum} 
        return {'h': attn_sum} 

    def apply_func(self, nodes):
        #norm = 1/(nodes.data['norm_sum']+1e-10)
        #return {'h': nodes.data['h'] * nodes.data['norm']}
        #return {'h': nodes.data['h']*norm}
        return {'h': nodes.data['h']}

    def propagate(self, g):
        g.update_all(self.msg_func, self.attn_reduce, self.apply_func)
        return g.ndata['h']

    def forward(self, g, h):

        # normalize relation attention
        self.weight_rel.data = torch.softmax(self.weight_rel.data, dim=0)

        g = g.local_var()
        g.ndata['h'] = h

        # self-loop pass
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight) + self.loop_bias_weight

        # edge update
        node_repr = self.propagate(g)

        # gating 1        
        if self.self_loop:
            self.gate_self_loop.data = torch.softmax(self.gate_self_loop.data, dim=0)

        
        # gating 2
        if node_repr.shape[1] == self.bias_weight.shape[0]:
            self.self_loop_att = torch.mm(torch.cat((loop_message, node_repr),dim=1), self.gating_attention)
            m = nn.Sigmoid()
            self.self_loop_att = m(self.self_loop_att)
            

        if node_repr.shape[1] != self.bias_weight.shape[0]:
            node_repr = loop_message
        elif self.bias:
            node_repr = node_repr + self.bias_weight
            # add self-loop + edge update
            if self.self_loop:
                # gating 1
                #node_repr = node_repr * self.gate_self_loop[0] + loop_message * self.gate_self_loop[1]
                
                # gating 2
                node_repr = node_repr * self.self_loop_att + loop_message * (1.0 - self.self_loop_att)
                #node_repr = node_repr + loop_message

        #node_repr = self.bn(node_repr)

        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h).view(node_repr.shape[0], self.out_feat)
            node_repr = node_repr + resval

        if self.activation:
            node_repr = self.activation(node_repr)

        node_repr = self.dropout(node_repr)

        return node_repr


class RWGCN_NET(nn.Module):
    def __init__(self, args):
        super(RWGCN_NET, self).__init__()

        self.num_nodes = args.num_nodes
        self.entity_feat_dim = args.entity_feat_dim
        self.num_edge_types=args.num_edge_types
        self.dropout = args.gnn_dropout

        n_hidden = args.decoder_embedding_dim
        n_layers = args.num_hidden - 1

        self.entity_embedding = nn.Embedding(self.num_nodes, self.entity_feat_dim, padding_idx=0).requires_grad_(False)

        self.rwgcn_layer = nn.ModuleList()

        #self.activation = F.tanh
        #self.activation = F.relu
        self.activation = nn.LeakyReLU(args.l_relu_ratio)

        # input projection
        self.rwgcn_layer.append(RWGCN_Layer(
            self.entity_feat_dim, n_hidden, num_edge_types=self.num_edge_types, activation=self.activation, dropout=self.dropout))

        for l in range(1, n_layers):
            self.rwgcn_layer.append(RWGCN_Layer(
                n_hidden, n_hidden, num_edge_types=self.num_edge_types, activation=self.activation, dropout=self.dropout))

        self.rwgcn_layer.append(RWGCN_Layer(
                n_hidden, n_hidden, num_edge_types=self.num_edge_types, activation=None, dropout=0.0))
        
        self.init()

    def init(self):

        init.xavier_uniform_(self.entity_embedding.weight)


    def en_regularization_loss(self):
        
        # no encoder regularization
        return 0

        reg_all_layer = 0
        for i, layer in enumerate(self.rwgcn_layer):
            reg_all_layer += layer.regularization_loss()
        return reg_all_layer

    def forward(self, g, node_id_copy):

        h = self.entity_embedding.weight[node_id_copy]

        for i, layer in enumerate(self.rwgcn_layer):
            h = layer(g, h)

        return h
        
        '''
        # perform max pooling over all layers
        all_layer_output = []
        for i, layer in enumerate(self.rwgcn_layer):
            h = layer(g, h)
            all_layer_output.append(h)

        # max-pooling
        all_layer_output = torch.stack(all_layer_output, axis=1)
        pooled_output = torch.max(all_layer_output,axis=1)[0]
        
        return pooled_output
        '''






# *****************************************************************
# Identity encoder

class identity_emb(nn.Module):
    def __init__(self, args):
        super(identity_emb, self).__init__()

        self.num_nodes = args.num_nodes
        self.entity_feat_dim = args.entity_feat_dim

        self.entity_embedding = nn.Embedding(self.num_nodes, self.entity_feat_dim, padding_idx=0).requires_grad_(False)
        self.init()

    def init(self):

        init.xavier_uniform_(self.entity_embedding.weight)

    def en_regularization_loss(self):
        return 0

    def forward(self, g, node_id_copy):

        h = self.entity_embedding.weight[node_id_copy]
        return h
        
# *****************************************************************