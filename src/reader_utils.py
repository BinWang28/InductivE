import torch
import numpy as np
import string


def get_vocab_idx(vocab, token):
    if token not in vocab:
        return vocab["UNK"]
    else:
        return vocab[token]


def get_relation_id(rel_name, train_network):
    rel_id = train_network.graph.find_relation(rel_name)
    if rel_id == -1: 
        return len(train_network.rel2id)
    else:
        return rel_id


# in data_loader.py
def prepare_batch_dgl(vocab, test_network, train_network):
    all_edges = []
    all_labels = []
    for edge in test_network.graph.iter_edges():
        src_id = get_vocab_idx(vocab, edge.src.name)
        tgt_id = get_vocab_idx(vocab, edge.tgt.name)
        rel_id = get_relation_id(edge.relation.name, train_network)
        all_edges.append(np.array([src_id, rel_id, tgt_id]))
        all_labels.append(edge.label)
    return np.array(all_edges), all_labels

# in train.py
def create_entity_dicts(all_tuples, num_rels, sim_relations=False):
    e1_to_multi_e2 = {}
    e2_to_multi_e1 = {}

    for tup in all_tuples:
        e1, rel, e2 = tup

        # No need to use sim edges for decoding
        if rel == num_rels-1 and sim_relations:
            continue

        rel_offset = num_rels

        if sim_relations:
            rel_offset -= 1

        if (e1, rel) in e1_to_multi_e2:
            e1_to_multi_e2[(e1, rel)].append(e2)
        else:
            e1_to_multi_e2[(e1, rel)] = [e2]

        if (e2, rel+rel_offset) in e1_to_multi_e2:
            e1_to_multi_e2[(e2, rel+rel_offset)].append(e1)
        else:
            e1_to_multi_e2[(e2, rel+rel_offset)] = [e1]

        if (e2, rel+rel_offset) in e2_to_multi_e1:
            e2_to_multi_e1[(e2, rel+rel_offset)].append(e1)
        else:
            e2_to_multi_e1[(e2, rel+rel_offset)] = [e1]
 
    return e1_to_multi_e2, e2_to_multi_e1

