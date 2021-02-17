import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
import torch.nn.functional as F

import encoder
from decoder import ConvTransE


class LinkPredictor(nn.Module):
    def __init__(self, args):
        super(LinkPredictor, self).__init__()
        
        # Parameters
        self.num_nodes = args.num_nodes
        #self.decoder_rels = args.num_rels * 2 # account for inverse relations
        self.decoder_rels = args.num_rels * 2 + 1 # account for inverse relations

        self.decoder_embedding_dim = args.decoder_embedding_dim
        self.rel_regularization = args.rel_regularization
        self.device = args.device

        self.encoder_name = args.encoder
        self.decoder_name = args.decoder

        # Entity & Rel Embedding
        self.w_relation = torch.nn.Embedding(self.decoder_rels, self.decoder_embedding_dim, padding_idx=0)
        self.entity_embedding = None
        
        # Encoder
        if self.encoder_name == 'RWGCN_NET':
            self.encoder = encoder.RWGCN_NET(args)
        elif self.encoder_name == 'identity_emb':
            self.encoder = encoder.identity_emb(args)
        else:
            print('Encoder not found: ', self.encoder_name)

        # Decoder
        if self.decoder_name == "ConvTransE":
            self.decoder = ConvTransE(self.num_nodes, self.decoder_rels, args)
        else:
            print('Decoder not found: ', self.decoder_name)

        #import pdb; pdb.set_trace()

        # Loss
        #self.loss = torch.nn.BCELoss()
        self.loss = torch.nn.BCELoss(reduction='none')
        #self.loss = torch.nn.BCELoss(reduction='mean')
        #self.loss = torch.nn.BCELoss(reduction='sum')

        #self.loss = torch.nn.KLDivLoss()
        #self.loss = torch.nn.KLDivLoss(reduction = 'sum')
        #self.loss = torch.nn.KLDivLoss(reduction = 'batchmean')
        self.init()

    def forward(self, e1, rel, target=None, sample_normaliaztion=None):

        e1_embedding = self.entity_embedding[e1]
        rel_embedding = self.w_relation(rel)

        # Decoder
        pred = self.decoder(e1_embedding, rel_embedding, self.entity_embedding)

        if target is None:
            return pred
        else:
            pred_loss = self.loss(pred, target)
            pred_loss = torch.mean(pred_loss, dim=1)
            
            if sample_normaliaztion != None:
                pred_loss = pred_loss * sample_normaliaztion
            pred_loss = torch.mean(pred_loss, dim=0)

            reg_loss = self.regularization_loss()
            if self.rel_regularization != 0.0:
                reg_loss = self.regularization_loss()
                return pred_loss + self.rel_regularization * reg_loss
            else:
                return pred_loss
            

    def init(self):

        xavier_normal_(self.w_relation.weight.data)

    def regularization_loss(self):
        
        reg_w_relation = self.w_relation.weight.pow(2)
        reg_w_relation = torch.mean(reg_w_relation)

        return reg_w_relation + self.encoder.en_regularization_loss()


    def update_whole_embedding_matrix(self, g, node_id_copy):
        
        if self.encoder_name == 'RWGCN_NET':
            gnn_embs = self.encoder.forward(g, node_id_copy)
            self.entity_embedding = gnn_embs
        elif self.encoder_name == 'identity_emb':
            #fixed_embs = self.encoder.forward(g, node_id_copy)
            #import pdb; pdb.set_trace()
            self.entity_embedding = self.encoder.entity_embedding.weight
        else:
            print('Encoder not found: ', self.encoder_name)