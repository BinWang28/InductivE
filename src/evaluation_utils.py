import numpy as np
import torch

import math
import json
import logging
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

#logging.basicConfig(filename='output.log', level=logging.INFO)

#######################################################################
# Utility functions for evaluation
#######################################################################


def sort_and_rank(score, target):
    sorted, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def get_filtered_ranks(score, target, batch_a, batch_r, e1_to_multi_e2):
    filtered_scores = score.clone().detach()

    for i, t in enumerate(target):
        filter_ents = e1_to_multi_e2[(batch_a[i].item(), batch_r[i].item())]

        # these filters contain ALL labels
        target_value = filtered_scores[i][t].clone()
        # zero all known cases => corresponds to the filtered setting
        filtered_scores[i][filter_ents] = 0.0
        assert t in filter_ents
        # write base the saved values
        filtered_scores[i][t] = target_value

    return sort_and_rank(filtered_scores, target)


def ranking_and_hits(args, model, test_triplets, e1_to_multi_e2, network, write_results=False, debug=False):
    #print(model)
    #test_triplets = test_triplets[:1200]

    s = test_triplets[:, 0]
    r = test_triplets[:, 1]
    o = test_triplets[:, 2]

    rel_offset = args.num_rels

    hits_left = []
    hits_right = []
    hits = []
    ranks = []
    ranks_left = []
    ranks_right = []
    scores = []
    node_mrr = {}

    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    batch_size = 64

    if debug:
        end = min(5000, len(test_triplets))
    else:
        end = len(test_triplets)
    
    #node_degree = []
    # for i in range(0, len(test_triplets), batch_size):
    for i in range(0, end, batch_size):
        e1 = s[i: i + batch_size]
        e2 = o[i: i + batch_size]
        rel = r[i: i + batch_size]
        rel_reverse = rel + rel_offset
        cur_batch_size = len(e1)

        e2_multi1 = [torch.LongTensor(e1_to_multi_e2[(e.cpu().item(), r.cpu().item())]) for e, r in zip(e1, rel)]
        e2_multi2 = [torch.LongTensor(e1_to_multi_e2[(e.cpu().item(), r.cpu().item())]) for e, r in
                     zip(e2, rel_reverse)]

        with torch.no_grad():
            #pred1 = model.calc_score(e1, rel)
            #pred2 = model.calc_score(e2, rel_reverse)
            pred1 = model(e1, rel)
            pred2 = model(e2, rel_reverse)

        pred1, pred2 = pred1.data.cpu(), pred2.data.cpu()
        scores.append(pred1)

        e1, e2 = e1.data, e2.data

        for j in range(0, cur_batch_size):
            # these filters contain ALL labels
            filter1 = e2_multi1[j].long()
            filter2 = e2_multi2[j].long()
            # save the prediction that is relevant
            target_value1 = pred1[j, e2[j].item()].item()
            target_value2 = pred2[j, e1[j].item()].item()
            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred1[j][filter1] = 0.0
            pred2[j][filter2] = 0.0

            # EXP: also remove self-connections
            pred1[j][e1[j].item()] = 0.0
            pred2[j][e2[j].item()] = 0.0

            # write base the saved values
            pred1[j][e2[j]] = target_value1
            pred2[j][e1[j]] = target_value2

        # sort and rank
        max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        max_values, argsort2 = torch.sort(pred2, 1, descending=True)

        for j in range(0, cur_batch_size):

            # find the rank of the target entities
            # rank1 = np.where(argsort1[i]==e2[i].item())[0][0]
            # rank2 = np.where(argsort2[i]==e1[i].item())[0][0]
            rank1 = (argsort1[j] == e2[j]).nonzero().cpu().item()
            rank2 = (argsort2[j] == e1[j]).nonzero().cpu().item()

            # rank+1, since the lowest rank is rank 1 not rank 0
            ranks.append(rank1 + 1)
            ranks_left.append(rank1 + 1)
            ranks.append(rank2 + 1)
            ranks_right.append(rank2 + 1)

            # record node degree
            #node_degree.append(degrees[e1[j].cpu().item()])
            #node_degree.append(degrees[e2[j].cpu().item()])

            node1 = network.graph.nodes[e1[j].cpu().item()]
            node2 = network.graph.nodes[e2[j].cpu().item()]

            if node1 not in node_mrr:
                node_mrr[node1] = []
            if node2 not in node_mrr:
                node_mrr[node2] = []

            node_mrr[node1].append(rank1)
            node_mrr[node2].append(rank2)

            for hits_level in range(0, 10):
                if rank1 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_left[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits[hits_level].append(1.0)
                    hits_right[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
                    hits_right[hits_level].append(0.0)
    ''' Logging changed to print
    for k in range(0, 10):
        logging.info('Hits left @{0}: {1}'.format(k + 1, np.mean(hits_left[k])))
        logging.info('Hits right @{0}: {1}'.format(k + 1, np.mean(hits_right[k])))
        logging.info('Hits @{0}: {1}'.format(k + 1, np.mean(hits[k])))
    logging.info('Mean rank left: {0}'.format(np.mean(ranks_left)))
    logging.info('Mean rank right: {0}'.format(np.mean(ranks_right)))
    logging.info('Mean rank: {0}'.format(np.mean(ranks)))
    logging.info('Mean reciprocal rank left: {0}'.format(np.mean(1. / np.array(ranks_left))))
    logging.info('Mean reciprocal rank right: {0}'.format(np.mean(1. / np.array(ranks_right))))
    logging.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))
    '''

    #for k in range(0, 10):
    for k in [0, 2, 9]:
        #print('Hits left @{0}: {1}'.format(k + 1, np.mean(hits_left[k])))
        #print('Hits right @{0}: {1}'.format(k + 1, np.mean(hits_right[k])))
        print('Hits @{0}: {1}'.format(k + 1, np.mean(hits[k])))
    #print('Mean rank left: {0}'.format(np.mean(ranks_left)))
    #print('Mean rank right: {0}'.format(np.mean(ranks_right)))
    print('Mean rank: {0}'.format(np.mean(ranks)))
    #print('Mean reciprocal rank left: {0}'.format(np.mean(1. / np.array(ranks_left))))
    #print('Mean reciprocal rank right: {0}'.format(np.mean(1. / np.array(ranks_right))))
    print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))


    if write_results:
        write_topk_tuples(torch.cat(scores, dim=0).cpu().numpy(), test_triplets, network)
        #plot_degree_mrr(node_mrr)
    

    return np.mean(1. / np.array(ranks))


def plot_degree_mrr(node_ranks):
    degree_rank = {}
    for node, rank in node_ranks.items():
        node_degree = node.get_degree()
        if node_degree not in degree_rank:
            degree_rank[node_degree] = []
        degree_rank[node_degree].append(sum(rank) / len(rank))

    degrees = []
    ranks = []
    for k in sorted(degree_rank.keys()):
        if k < 20:
            # degrees.append(k)
            # ranks.append(sum(degree_rank[k])/len(degree_rank[k]))
            for rank in degree_rank[k]:
                if rank < 100:
                    degrees.append(k)
                    ranks.append(rank)

    fig, ax = plt.subplots()

    ax.scatter(degrees, ranks, marker='.')
    ax.set(xlabel="degree", ylabel="mean ranks")
    ax.grid()
    fig.savefig("comet_cn_degree_ranks.png")

    # In degree
    avg_mrr = {}
    for k in sorted(degree_rank.keys()):
        #print(len(degree_rank[k]))
        if len(degree_rank[k]) >= 10:
            print(k, ': ', len(degree_rank[k]))
            avg_mrr[k] = np.mean(1.0 / (np.array(degree_rank[k])+1))
    print(avg_mrr)
    d_l, r_l = [], []
    for d, r in avg_mrr.items():
        d_l.append(d)
        r_l.append(r)

    fig, ax = plt.subplots()
    ax.bar(d_l, r_l)

    ax.set(xlabel='in-degree', ylabel='MRR',
        title='x')
    ax.grid()

    fig.savefig("val2.png") 




def write_topk_tuples(scores, input_prefs, network, k=5):
    out_lines = []

    argsort = [np.argsort(-1 * np.array(score)) for score in np.array(scores)]

    for i, sorted_scores in enumerate(argsort):

        pref = input_prefs[i]
        e1 = pref[0].cpu().item()
        rel = pref[1].cpu().item()
        e2 = pref[2].cpu().item()
        cur_point = {}
        cur_point['gold_triple'] = {}
        cur_point['gold_triple']['e1'] = network.graph.nodes[e1].name
        cur_point['gold_triple']['e2'] = network.graph.nodes[e2].name
        cur_point['gold_triple']['relation'] = network.graph.relations[rel].name

        topk_indices = sorted_scores[:k]
        topk_tuples = [network.graph.nodes[elem] for elem in topk_indices]
        # if golds[i] in topk_tuples:
        #    topk_indices = argsort[i][:k+1]
        #    topk_tuples = [input_batch[i][elem] for elem in topk_indices if input_batch[i][elem]!=golds[i]]
        cur_point['candidates'] = []

        for j, node in enumerate(topk_tuples):
            tup = {}
            tup['e1'] = network.graph.nodes[e1].name
            tup['e2'] = node.name
            tup['relation'] = network.graph.relations[rel].name
            tup['score'] = str(scores[i][topk_indices[j]])
            cur_point['candidates'].append(tup)

        out_lines.append(cur_point)

    with open("topk_candidates.jsonl", 'w') as f:
        for entry in out_lines:
            json.dump(entry, f)
            f.write("\n")
