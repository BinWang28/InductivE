import os
import sys
import dgl
import pickle

import random
import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import coo_matrix
from scipy import sparse

torch.set_printoptions(profile="full")



def get_adj_and_degrees(num_nodes, num_rels, triplets):
    """ Get adjacency list and degrees of the graph
    """
    
    col = []
    row = []
    rel = []
    adj_list = [[] for _ in range(num_nodes)]
    for i, triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

        # in-degree adj
        #adj_list[triplet[2]].append([triplet[0], triplet[1]])
        #adj_list[triplet[0]].append([triplet[2], triplet[1] + num_rels])
        
        
        row.append(triplet[0])
        col.append(triplet[2])
        rel.append(triplet[1])
        row.append(triplet[2])
        col.append(triplet[0])
        rel.append(triplet[1] + num_rels)

    sparse_adj_matrix = coo_matrix((np.ones(len(triplets)*2), (row, col)), shape=(num_nodes, num_nodes))
   
    degrees = np.array([len(a) for a in adj_list]) # sum of in and out degree
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees, sparse_adj_matrix, rel



def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def load_pre_computed_feat(feat_path, feat_dim, id2node):
    '''compute bert feature from dictionary
    '''
    # Load embedding dict
    with open(feat_path, 'rb') as fp:
        node_emb_dict = pickle.load(fp)

    load_dim = node_emb_dict[random.choice(list(node_emb_dict))].shape[0]

    node_embed = np.zeros((len(id2node), load_dim))
    for i in range(len(id2node)):
        
        # find match
        if id2node[i] in node_emb_dict:
            node_embed[i] = node_emb_dict[id2node[i]]
        else:
            node_embed[i] = node_emb_dict[id2node[i].lower()]

    print("Loaded Feature from: ", feat_path)
    print("Loaded Feature Shape: ", node_embed.shape)

    # lower the dimension if necessary
    if feat_dim > node_embed.shape[1]:
        print("Desired dimension larger than loaded embedding. Please check!\n")
        sys.exit()
    elif feat_dim < node_embed.shape[1]:
        print("BERT desired dimension smaller than loaded embedding. Perform PCA...")
        pca = PCA(n_components=feat_dim)
        node_embed = pca.fit_transform(node_embed)
        print("New Embedding Shape:", node_embed.shape) 

    node_embed = torch.tensor(node_embed)

    return node_embed


def aggregrate(bert_feature, adj_list):

    print('Aggregrate neighboring features...')

    adj_tail = []
    for neighbor in adj_list:
        if neighbor.shape[0] == 0:
            adj_tail.append([])
        else:
            adj_tail.append(neighbor[:,1].tolist())
    

    new_bert_fea = []
    for i in range(len(adj_list)):

        bert_feature[adj_tail[i]]
        bert_feature[i].unsqueeze(0)

        entity_fea = torch.cat((bert_feature[i].unsqueeze(0), bert_feature[adj_tail[i]])).mean(0)
        new_bert_fea.append(entity_fea)
    
    return torch.stack(new_bert_fea, axis=0)




def sample_sub_graph(args, sample_size, tri_graph): # sample nodes
    ''' Sample a sub graph with new index
    '''

    num_nodes = args.num_nodes
    if sample_size >= num_nodes:
        sample_size = num_nodes
    uniq_v = np.random.choice(num_nodes, sample_size, replace=False)
    uniq_v = np.sort(uniq_v)
    
    src = []
    dst = []
    edge_type = []

    tri_edge_src, tri_edge_tgt, tri_edge_type = tri_graph

    filtered_edges = []
    for i in range(len(tri_edge_src)):
        head = tri_edge_src[i]
        rel = tri_edge_type[i]
        tail = tri_edge_tgt[i]
        if head in uniq_v and tail in uniq_v:
            src.append(head)
            dst.append(tail)
            edge_type.append(rel)
            #if len(src) >= 10:
            #    break


    edge_type = np.array(edge_type)
    _, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))

    # build DGL graph
    g = dgl.DGLGraph()
    g.add_nodes(len(uniq_v))
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    print("# nodes: {}, # edges: {}".format(len(uniq_v), len(src)))

    # update graph information
    node_id = uniq_v
    node_norm = norm

    node_id_copy = np.copy(node_id)
    node_dict = {v: k for k, v in dict(enumerate(node_id_copy)).items()}
    # set node/edge feature
    node_id = torch.from_numpy(node_id).view(-1, 1)
    node_norm = torch.from_numpy(node_norm).view(-1, 1)
    edge_type = torch.from_numpy(edge_type)

    node_id = node_id.to(args.device)
    node_norm = node_norm.to(args.device)
    edge_type = edge_type.to(args.device)

    g.ndata.update({'id': node_id, 'norm': node_norm})
    g.edata['type'] = edge_type

    return g, uniq_v, norm




def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm


import timeit
def dynamic_graph_gen(args, entity_embedding, n_ontology=1, inductive_index=[], degrees=[]):
    ''' update graph from entity embedding by global thresholding
    '''

    if n_ontology < 1:
        print('\n**************************')
        print('Perform global thresholding for graph generation')
        start_time = timeit.default_timer()

        threshold = n_ontology
        num_nodes = args.num_nodes

        sim_edge_src = []
        sim_edge_tgt = []
        sim_edge_type = []
        
        batch_size = 1000

        for row_i in range(0, int(entity_embedding.shape[0] / batch_size) + 1):

            start = row_i * batch_size
            end = min([(row_i + 1) * batch_size, entity_embedding.shape[0]])
            if end <= start:
                break
            rows = entity_embedding[start: end]
            sim = cosine_similarity(rows, entity_embedding)  

            # set diagonal matrix to zero, avoiding count self-loop
            for i in range(end-start):
                ind = i + start
                sim[i, ind] = 0

            # set neighboring entities
            #sim[sim < threshold] = 0
            #sim[sim >= threshold] = 1

            sim_edge_src.extend((np.where(sim >= threshold)[0]+start).tolist())
            sim_edge_tgt.extend(np.where(sim >= threshold)[1].tolist())

            #result = np.where(sim==1)[0].tolist()
            #sim_edge_src.extend(np.where(sim==1)[0].tolist())
            #sim_edge_tgt = np.where(sim==1)[1]

        sim_edge_type = [args.num_edge_types-1] * len(sim_edge_src)
        print('Number of semantic similarity edges: ', len(sim_edge_src))
        stop_time = timeit.default_timer()
        print('Time: ', stop_time - start_time)
        print('**************************')
    
    else:

        print('\n**************************')
        print('knn graph for graph generation')
        start_time = timeit.default_timer()
        
        threshold = int(n_ontology)
        num_nodes = args.num_nodes

        sim_edge_src = []
        sim_edge_tgt = []
        sim_edge_type = []
        
        batch_size = 1000

        for row_i in range(0, int(entity_embedding.shape[0] / batch_size) + 1):

            start = row_i * batch_size
            end = min([(row_i + 1) * batch_size, entity_embedding.shape[0]])
            if end <= start:
                break
            rows = entity_embedding[start: end]
            sim = cosine_similarity(rows, entity_embedding)  

            # set diagonal matrix to zero, avoiding count self-loop
            for i in range(end-start):
                ind = i + start
                sim[i, ind] = 0

            for i in range(sim.shape[0]):
                if (threshold-degrees[i+start]) <= 0:
                    continue
                indexing = np.argsort(sim[i])[-(threshold-degrees[i+start]):]
                for j in range(indexing.shape[0]):
                    src = indexing[j]
                    sim_edge_src.append(src)
                    sim_edge_tgt.append(i+start)


            ''' # old way for knn
            # Find index from similarity measure (time consuming for ranking)
            sorted_row_idx = np.argsort(sim, axis=1)[:,sim.shape[1]-(threshold)::]
            col_idx = np.arange(sim.shape[0])[:,None]
            
            
            # set neighboring entities
            sim[:,:] = 0 # Set all items to zero
            sim[col_idx, sorted_row_idx] = 1

            sim_edge_src.extend((np.where(sim == 1)[0]+start).tolist())
            sim_edge_tgt.extend((np.where(sim == 1)[1]).tolist())

            # only process one batch for debugging purpose
            #if len(sim_edge_tgt) > batch_size:
            #    break
            '''

        sim_edge_type = [args.num_edge_types-1] * len(sim_edge_src)
        print('Number of semantic similarity edges: ', len(sim_edge_src))
        stop_time = timeit.default_timer()
        print('Time: ', stop_time - start_time)
        print('**************************')

    print('Number of NA need to be filtered: ', len(inductive_index))

    filtered_sim_edge_src = []
    filtered_sim_edge_tgt = []
    filtered_sim_edge_type = []

    for i in range(len(sim_edge_src)):
        if sim_edge_src[i] not in inductive_index and sim_edge_tgt[i] not in inductive_index:
            filtered_sim_edge_src.append(sim_edge_src[i])
            filtered_sim_edge_tgt.append(sim_edge_tgt[i])
            filtered_sim_edge_type.append(sim_edge_type[i])
    
    print('Number of similarity edges after filtering: ', len(filtered_sim_edge_src))
    print('**************************')

    return filtered_sim_edge_src, filtered_sim_edge_tgt, filtered_sim_edge_type





def create_triplet_graph(args, train_data):
    ''' create graph from triplets
    '''

    num_nodes = args.num_nodes
    num_rels = args.num_rels

    edge_src = []
    tri_edge_type = []
    edge_tgt = []

    for tup in train_data:
        e1, rel, e2 = tup
        edge_src.append(e1)
        edge_tgt.append(e2)
        tri_edge_type.append(rel)

        edge_src.append(e2)
        edge_tgt.append(e1)
        tri_edge_type.append(rel+num_rels)

        #if e1 == e2:
        #    import pdb; pdb.set_trace()
    print('# Triplet graph edges: ', len(edge_src))

    return edge_src, edge_tgt, tri_edge_type



from tqdm import trange
import textdistance


def levenshtein_sim_graph(args, id2node, n_ontology):

    node_list = [v for k,v in id2node.items()]
    #node_list = node_list[:1000]
    num_nodes = len(node_list)

    sim_edge_src = []
    sim_edge_tgt = []
    sim_edge_type = []

    batch_size = 1000
    for row_i in range(0, int(num_nodes / batch_size) + 1):
        start = row_i * batch_size
        end = min([(row_i + 1) * batch_size, num_nodes])
        if end <= start:
            break

        for i in range(start, end):
            for j in range(num_nodes):
                
                if i != j:    
                    ww_dist = textdistance.levenshtein.normalized_similarity(node_list[i], node_list[j])

                    if ww_dist >= n_ontology:
                        sim_edge_src.append(i)
                        sim_edge_tgt.append(j)


    sim_edge_type = [args.num_edge_types-1] * len(sim_edge_src)
    print('Number of levenshtein similarity edges: ', len(sim_edge_src))

    print('*********************')
    print('Examples from levenshtein')
    if len(sim_edge_src)>10:
        for i in range(10):
            print(node_list[sim_edge_src[i]], '===', node_list[sim_edge_tgt[i]])
    print('*********************')

    return sim_edge_src, sim_edge_tgt, sim_edge_type




import matplotlib.pyplot as plt
def utils_plot_dist(args, train_data, valid_data, test_data):

    all_data = train_data + valid_data + test_data
    degrees = [0] * args.num_nodes

    for triplet in all_data:
        degrees[triplet[-1]] += 1
        degrees[triplet[0]] += 1

    '''
    #train_data = random.choices(train_data, k=1200)
    #train_data = random.choices(train_data, k=10240)

    train_entities = []
    for triplet in train_data:
        if triplet[0] not in train_entities:
            train_entities.append(triplet[0])
        if triplet[-1] not in train_entities:
            train_entities.append(triplet[-1])
    train_degrees = [degrees[item] for item in train_entities]

    valid_entities = []
    for triplet in valid_data:
        if triplet[0] not in valid_entities:
            valid_entities.append(triplet[0])
        if triplet[-1] not in valid_entities:
            valid_entities.append(triplet[-1])
    valid_degrees = [degrees[item] for item in valid_entities]

    test_entities = []
    for triplet in test_data:
        if triplet[0] not in test_entities:
            test_entities.append(triplet[0])
        if triplet[-1] not in test_entities:
            test_entities.append(triplet[-1])
    test_degrees = [degrees[item] for item in test_entities]

    train_degrees = np.array(train_degrees)
    valid_degrees = np.array(valid_degrees)
    test_degrees = np.array(test_degrees)

    size = 4

    train_ratio = []
    train_ratio.append(((0 <= train_degrees) & (train_degrees < 2)).sum())
    train_ratio.append(((2 <= train_degrees) & (train_degrees < 3)).sum())
    train_ratio.append(((3 <= train_degrees) & (train_degrees < 15)).sum())
    train_ratio.append(((15 <= train_degrees) & (train_degrees < 100000)).sum())
    train_ratio = np.array(train_ratio)
    train_ratio = train_ratio / len(train_degrees)

    valid_ratio = []
    valid_ratio.append(((0 <= valid_degrees) & (valid_degrees < 2)).sum())
    valid_ratio.append(((2 <= valid_degrees) & (valid_degrees < 3)).sum())
    valid_ratio.append(((3 <= valid_degrees) & (valid_degrees < 15)).sum())
    valid_ratio.append(((15 <= valid_degrees) & (valid_degrees < 100000)).sum())
    valid_ratio = np.array(valid_ratio)
    valid_ratio = valid_ratio / len(valid_degrees)

    test_ratio = []
    test_ratio.append(((0 <= test_degrees) & (test_degrees < 2)).sum())
    test_ratio.append(((2 <= test_degrees) & (test_degrees < 3)).sum())
    test_ratio.append(((3 <= test_degrees) & (test_degrees < 15)).sum())
    test_ratio.append(((15 <= test_degrees) & (test_degrees < 100000)).sum())
    test_ratio = np.array(test_ratio)
    test_ratio = test_ratio / len(test_degrees)

    x = np.arange(size)
    total_width, n = 0.8, 3
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.bar(x, train_ratio,  width=width, label='train')
    plt.bar(x + width, valid_ratio, width=width, label='val', tick_label=['deg=1','2<=deg<3','3<=deg<15','15<=deg'])
    plt.bar(x + 2 * width, test_ratio, width=width, label='test')
    plt.xlabel('entity degree range')
    plt.ylabel('percentage')
    plt.legend()
    plt.show()
    plt.savefig("cn82_degree.pdf", bbox_inches='tight') 

    #train_ratio.append(sum(train_degrees == 2)/len(train_degrees))
    #train_ratio.append(sum(train_degrees == 3)/len(train_degrees))
    #train_ratio.append(sum(train_degrees == 4)/len(train_degrees))
    '''

    

    train_degrees = []
    for triplet in train_data:
        ave_edge_degree = (degrees[triplet[0]] + degrees[triplet[-1]]) / 2
        train_degrees.append(ave_edge_degree)
    
    valid_degrees = []
    for triplet in valid_data:
        ave_edge_degree = (degrees[triplet[0]] + degrees[triplet[-1]]) / 2
        valid_degrees.append(ave_edge_degree)
    
    test_degrees = []
    for triplet in test_data:
        ave_edge_degree = (degrees[triplet[0]] + degrees[triplet[-1]]) / 2
        test_degrees.append(ave_edge_degree)

    all_degrees = []
    for triplet in all_data:
        ave_edge_degree = (degrees[triplet[0]] + degrees[triplet[-1]]) / 2
        all_degrees.append(ave_edge_degree)
    

    train_degrees = np.array(train_degrees)
    valid_degrees = np.array(valid_degrees)
    test_degrees = np.array(test_degrees)
    all_degrees = np.array(all_degrees)

    size = 4

    train_ratio = []
    train_ratio.append(((0 <= train_degrees) & (train_degrees < 3)).sum())
    train_ratio.append(((3 <= train_degrees) & (train_degrees < 15)).sum())
    train_ratio.append(((15 <= train_degrees) & (train_degrees < 35)).sum())
    train_ratio.append(((35 <= train_degrees) & (train_degrees < 100000)).sum())
    train_ratio = np.array(train_ratio)
    train_ratio = train_ratio / len(train_degrees)

    valid_ratio = []
    valid_ratio.append(((0 <= valid_degrees) & (valid_degrees < 3)).sum())
    valid_ratio.append(((3 <= valid_degrees) & (valid_degrees < 15)).sum())
    valid_ratio.append(((15 <= valid_degrees) & (valid_degrees < 35)).sum())
    valid_ratio.append(((35 <= valid_degrees) & (valid_degrees < 100000)).sum())
    valid_ratio = np.array(valid_ratio)
    valid_ratio = valid_ratio / len(valid_degrees)

    test_ratio = []
    test_ratio.append(((0 <= test_degrees) & (test_degrees < 3)).sum())
    test_ratio.append(((3 <= test_degrees) & (test_degrees < 15)).sum())
    test_ratio.append(((15 <= test_degrees) & (test_degrees < 35)).sum())
    test_ratio.append(((35 <= test_degrees) & (test_degrees < 100000)).sum())
    test_ratio = np.array(test_ratio)
    test_ratio = test_ratio / len(test_degrees)

    all_ratio = []
    all_ratio.append(((0 <= all_degrees) & (all_degrees < 3)).sum())
    all_ratio.append(((3 <= all_degrees) & (all_degrees < 15)).sum())
    all_ratio.append(((15 <= all_degrees) & (all_degrees < 35)).sum())
    all_ratio.append(((35 <= all_degrees) & (all_degrees < 100000)).sum())
    all_ratio = np.array(all_ratio)
    all_ratio = all_ratio / len(all_degrees)

    x = np.arange(size)
    total_width, n = 0.8, 4
    width = total_width / n
    x = x - (total_width - width) / 2

    #plt.bar(x, all_ratio, width=width, label='all')
    plt.bar(x + 0*width, train_ratio,  width=width, label='train')
    #plt.bar(x + 2* width, valid_ratio, width=width, label='val', tick_label=['deg<3','3<=deg<15','15<=deg<35','35<=deg'])
    plt.bar(x + 1* width, valid_ratio, width=width, label='val')

    plt.bar(x + 2 * width, test_ratio, width=width, label='test')

    plt.xticks(x+1.0*width, labels = ['[0,3)','[3,15)','[15,35)','[35,inf)'],fontsize=20)
    #plt.rc('axes', titlesize=200) 
    #plt.rc('ytick',labelsize=200)
    plt.yticks(fontsize=20)

    plt.rcParams['font.size'] = 18

    plt.xlabel('triplet-degree range',fontsize=22)
    plt.ylabel('percentage of triplets',fontsize=22)
    plt.legend()
    plt.show()
    plt.savefig("cn82_triplet_degree.png", bbox_inches='tight') 
    plt.savefig("cn82_triplet_degree.pdf", bbox_inches='tight') 

    #train_ratio.append(sum(train_degrees == 2)/len(train_degrees))
    #train_ratio.append(sum(train_degrees == 3)/len(train_degrees))
    #train_ratio.append(sum(train_degrees == 4)/len(train_degrees))



    import pdb; pdb.set_trace()


    # test
    size = 4
    x = np.arange(size)
    a = np.random.random(size)
    b = np.random.random(size)
    c = np.random.random(size)

    total_width, n = 0.8, 3
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.bar(x, a,  width=width, label='train')
    plt.bar(x + width, b, width=width, label='valid')
    plt.bar(x + 2 * width, c, width=width, label='test')
    plt.legend()
    plt.show()
    plt.savefig("val2.png") 

    '''
    # plots
    fig, ax = plt.subplots()
    ax.scatter(test_degrees, test_degrees, marker='.')
    ax.set(xlabel="degree", ylabel="mean ranks")
    ax.grid()
    fig.savefig("comet_cn_degree_ranks.png")
    fig.savefig("val2.png") 
    '''

    import pdb; pdb.set_trace()



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
