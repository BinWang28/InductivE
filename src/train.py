import os
import sys
import time
import random
from copy import deepcopy

import torch
import numpy as np

# Ours
import parse_args
import data_loader
import reader_utils
import utils
import evaluation_utils

from model import LinkPredictor


def main(args):

    # set random seed
    utils.set_seeds(args.seed)
    # check cuda
    use_cuda = torch.cuda.is_available()
    if use_cuda and not args.no_cuda:
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")




    # - - - - - - - - - - - - - - - - - Data Loading - - - - - - - - - - - - - - - - - - -
    # Load train data only
    train_data, valid_data, test_data, train_network, id2node = data_loader.load_data(args)
    num_nodes = len(train_network.graph.nodes)
    num_rels = len(train_network.graph.relations)
    args.num_nodes = num_nodes
    args.num_rels = num_rels
    
    # testing atomic using a subset for fater evaluation
    if args.dataset == 'atomic':
        # only subset for speed
        valid_data = valid_data[:10000]
        test_data = test_data[:10000]
    
    print('Train Triplet #: ', train_data.shape[0], end = '  ')
    print('Val Triplet #: ', valid_data.shape[0], end = '  ')
    print('Test Triplet #: ', test_data.shape[0])  

    # Plot data distribution
    #utils.utils_plot_dist(args, train_data.tolist(), valid_data.tolist(), test_data.tolist())


    # calculate degrees for entities
    _, degrees, _, _ = utils.get_adj_and_degrees(num_nodes, num_rels, train_data)

    # Filter validation and testing and only keep the NA Triplets nodes
    inductive_index = (np.where(degrees == 0)[0]).tolist()

    filtered_valid = []
    filtered_test = []

    for item in valid_data:
        if degrees[item[0]] == 0 or degrees[item[2]] == 0:
            filtered_valid.append(item)
    for item in test_data:
        if degrees[item[0]] == 0 or degrees[item[2]] == 0:
            filtered_test.append(item)

    filtered_valid = np.array(filtered_valid)
    filtered_test = np.array(filtered_test)

    print('NA Val Triplet #: ', filtered_valid.shape[0], end = '  ')
    print('NA Test Triplet #: ', filtered_test.shape[0])



    # - - - - - - - - - - - - - - - - - Statistics for evaluation - - - - - - - - - - - - - - - - - - -
    all_tuples = train_data.tolist() + valid_data.tolist() + test_data.tolist()
    # for filtered ranking (for evaluation)
    all_e1_to_multi_e2, all_e2_to_multi_e1 = reader_utils.create_entity_dicts(all_tuples, num_rels)
    # for training (used for computing 1-M Graph)
    train_e1_to_multi_e2, train_e2_to_multi_e1 = reader_utils.create_entity_dicts(train_data.tolist(), num_rels)
    
    filtered_valid = torch.LongTensor(filtered_valid)
    filtered_test = torch.LongTensor(filtered_test)
    filtered_valid = filtered_valid.to(args.device)
    filtered_test = filtered_test.to(args.device)

    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)
    valid_data = valid_data.to(args.device)
    test_data = test_data.to(args.device)
 


    # - - - - - - - - - - - - - - - - - Pre-trained feature loading - - - - - - - - - - - - - - - - - - -

    # Embedding initialization
    if args.bert_feat_path != 'None' and args.fasttext_feat_path != 'None':
        bert_feature = utils.load_pre_computed_feat(args.bert_feat_path, args.bert_feat_dim, id2node)    
        fasttext_feature = utils.load_pre_computed_feat(args.fasttext_feat_path, args.fasttext_feat_dim, id2node)    
        fusion_feature = torch.cat((bert_feature, fasttext_feature),dim=1)
        print("Loading Pre-computed BERT and fasttext Embedding")
    elif args.bert_feat_path != 'None':
        bert_feature = utils.load_pre_computed_feat(args.bert_feat_path, args.bert_feat_dim, id2node)    
        print("Loading Pre-computed BERT Embedding")
    elif args.fasttext_feat_path != 'None':
        fasttext_feature = utils.load_pre_computed_feat(args.fasttext_feat_path, args.fasttext_feat_dim, id2node)    
        print("Loading Pre-computed fasttext Embedding")
    else:
        print("No node feature provided. Use random initialization")
    print('')




    # - - - - - - - - - - - - - - - - - Fixed Graph Preparation - - - - - - - - - - - - - - - - - - -
    # Fixed graph
    fix_edge_src = []
    fix_edge_tgt = []
    fix_edge_type = []

    args.num_edge_types = 0

    # create a triplet graph (with edge types)
    if args.fix_triplet_graph:
        tri_edge_src, tri_edge_tgt, tri_edge_type = utils.create_triplet_graph(args, train_data.tolist())
        tri_graph = (tri_edge_src, tri_edge_tgt, tri_edge_type)
        print('Number of triplet edges: ', len(tri_edge_src))
        print('Number of triplet edges types: ', args.num_rels * 2)
        print('')

        fix_edge_src.extend(tri_graph[0])
        fix_edge_tgt.extend(tri_graph[1])
        fix_edge_type.extend(tri_graph[2])

    if args.fix_triplet_graph:
        args.num_edge_types = args.num_edge_types + args.num_rels * 2    

    # Add similarity graph edge
    if args.dynamic_sim_graph:
        print('Add sim edge type for semantic similarity graph')
        args.num_edge_types = args.num_edge_types + 1
    
    print('Number of relation types for R-GCN model: ', args.num_edge_types)
    

    fixed_graph = (fix_edge_src, fix_edge_tgt, fix_edge_type)
    print('Total number of fixed edges: ', len(fix_edge_src))
    print('')




    # - - - - - - - - - - - - - - - - - Model Initialization - - - - - - - - - - - - - - - - - - -

    # create model
    model = LinkPredictor(args)
    print(model)

    # embedding initialization
    if args.bert_feat_path != 'None' and args.fasttext_feat_path != 'None':
        print("Initialize with concatenated BERT and fastText Embedding")
        model.encoder.entity_embedding.load_state_dict({'weight': fusion_feature})
    elif args.bert_feat_path != 'None':
        print("Initialize with Pre-computed BERT Embedding")
        model.encoder.entity_embedding.load_state_dict({'weight': bert_feature})
    elif args.fasttext_feat_path != 'None':
        print("Initialize with Pre-computed fasttext Embedding")
        model.encoder.entity_embedding.load_state_dict({'weight': fasttext_feature})
    else:
        print("No node feature provided. Use uniform initialization")
    
    model.to(args.device)


    # - - - - - - - - - - - - - - Evaluation Only - - - - - - - - - - - - - - - - - - -
    # TODO, not finished
    if args.eval_only:
        if args.load_model:
            model_state_file = args.load_model
        else:
            print("Please provide model path for evaluation (--load_model)")
            sys.exit(0)

        checkpoint = torch.load(model_state_file)

        model.eval()
        model.load_state_dict(checkpoint['state_dict'])
        #print(model)

        eval_graph = checkpoint['eval_graph']
        print("Using best epoch: {}".format(checkpoint['epoch']))

        # Update whole graph embedding
        g_whole, node_id, node_norm = utils.sample_sub_graph(args, 1000000000, eval_graph)
        if args.dataset == 'atomic' or args.dataset[:10] == 'conceptnet':
            print('perform evaluation on cpu')
            model.cpu()
            g_whole.ndata['id'] = g_whole.ndata['id'].cpu()
            g_whole.ndata['norm'] = g_whole.ndata['norm'].cpu()
            g_whole.edata['type'] = g_whole.edata['type'].cpu()
            valid_data = valid_data.cpu()
            test_data = test_data.cpu()
            filtered_valid = filtered_valid.cpu()
            filtered_test = filtered_test.cpu()

        # update all embedding
        if model.entity_embedding != None:
            del model.entity_embedding
            model.entity_embedding = None
            torch.cuda.empty_cache()
        node_id_copy = np.copy(node_id)
        model.update_whole_embedding_matrix(g_whole, node_id_copy)

        print("===========DEV============")
        evaluation_utils.ranking_and_hits(args, model, valid_data, all_e1_to_multi_e2, train_network)
        print("================TEST================")
        evaluation_utils.ranking_and_hits(args, model, test_data, all_e1_to_multi_e2, train_network, write_results=False)

        sys.exit(0)

        '''
        # test similarity edges
        with open('record_sim.txt', 'a') as f:
            for i in range(len(eval_graph[0])):
                f.write(str(eval_graph[1][i]))
                f.write(': ')
                f.write(id2node[eval_graph[1][i]])
                f.write('  ')
                f.write(str(eval_graph[0][i]))
                f.write(': ')
                f.write(id2node[eval_graph[0][i]])
                f.write('\n')
        sys.exit(0)
        '''

        # Bin
        # Update whole graph embedding
        g_whole, node_id, node_norm = utils.sample_sub_graph(args, 99999999999999, current_graph)
        node_id_copy = np.copy(node_id)
        model.update_whole_embedding_matrix(g_whole, node_id_copy)

        print("================DEV=================")
        mrr = evaluation_utils.ranking_and_hits(args, model, valid_data, all_e1_to_multi_e2, train_network, write_results=True)

        print("================TEST================")
        mrr = evaluation_utils.ranking_and_hits(args, model, test_data, all_e1_to_multi_e2, train_network, write_results=True)

        sys.exit(0)



    # - - - - - - - - - - - - - - - - - Start Training - - - - - - - - - - - - - - - - - - -

    t = time.localtime()
    cur_time = time.strftime("%b_%d_%H_%M_%S", t)
    args.model_state_file = os.path.join(args.output_dir, cur_time + '.pt')
    print("Model Save Path: ", args.model_state_file)

    # Current Model Exists
    if os.path.isfile(args.model_state_file):
        print(args.model_state_file)
        if not args.overwrite:
            print('Model already exists. Use Overwrite')
            sys.exit(0)

    # optimizer
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.regularization)
        #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3)

    elif args.optimizer == "Adagrad":
        optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    #import pdb; pdb.set_trace()
    forward_time = []
    backward_time = []

    # training loop
    print("Starting training...")
    epoch = 0
    best_mrr = 0
    f_best_mrr = 0
    patient_times = 0
    batch_size = args.decoder_batch_size

    current_graph = fixed_graph
    fixed_train_data = train_data[:]
    print('Number of fixed training data: ', fixed_train_data.shape[0])

    while True:
        epoch += 1

        # - - - - - - - - - - - - - - Dynamic Graph Generation - - - - - - - - - - - - - - - - - - -
        if args.dataset == 'atomic' and epoch % args.evaluate_every == 1 and epoch != 1:
            # Update after evaluation
            degrees_copy = deepcopy(degrees)
            sim_edge_src, sim_edge_tgt, sim_edge_type = utils.dynamic_graph_gen(
                                                                args, 
                                                                model.entity_embedding.detach().cpu().numpy(), 
                                                                n_ontology=args.n_ontology, 
                                                                inductive_index=inductive_index,
                                                                degrees=degrees_copy
                                                                )

            sim_graph = (sim_edge_src, sim_edge_tgt, sim_edge_type)
            
            
            # Merge two graphs
            cur_edge_src = []
            cur_edge_tgt = []
            cur_edge_type = []

            cur_edge_src.extend(fixed_graph[0])
            cur_edge_tgt.extend(fixed_graph[1])
            cur_edge_type.extend(fixed_graph[2])

            cur_edge_src.extend(sim_graph[0])
            cur_edge_tgt.extend(sim_graph[1])
            cur_edge_type.extend(sim_graph[2])

            current_graph = (cur_edge_src, cur_edge_tgt, cur_edge_type)

            print('Number of edges in augmented graph with sim and syn_triplets: ', len(current_graph[0]))

        
        if args.dataset[:10] == 'conceptnet' and args.dynamic_sim_graph and epoch > args.start_dynamic_graph and epoch % args.dynamic_graph_ee_epochs == 1:
            print('')
            print("**************** Update Graph *******************************")

            print('Update dynamic similarity graph')
            
            cur_edge_src = []
            cur_edge_tgt = []
            cur_edge_type = []

            cur_edge_src.extend(fixed_graph[0])
            cur_edge_tgt.extend(fixed_graph[1])
            cur_edge_type.extend(fixed_graph[2])

            # compute whole graph embedding
            g_whole, node_id, node_norm = utils.sample_sub_graph(args, 99999999999999, current_graph)
            node_id_copy = np.copy(node_id)
            if model.entity_embedding != None:
                del model.entity_embedding
                model.entity_embedding = None

            model.eval()
            if args.dataset == 'atomic' or args.dataset[:10] == 'conceptnet':
                # perform evaluation on cpu
                model.cpu()
                g_whole.ndata['id'] = g_whole.ndata['id'].cpu()
                g_whole.ndata['norm'] = g_whole.ndata['norm'].cpu()
                g_whole.edata['type'] = g_whole.edata['type'].cpu()
            
            torch.cuda.empty_cache()
            model.update_whole_embedding_matrix(g_whole, node_id_copy)

            # similarity graph construction
            degrees_copy = deepcopy(degrees)
            sim_edge_src, sim_edge_tgt, sim_edge_type = utils.dynamic_graph_gen(
                                                                args, 
                                                                model.entity_embedding.detach().cpu().numpy(), 
                                                                n_ontology=args.n_ontology, 
                                                                inductive_index=inductive_index,
                                                                degrees=degrees_copy
                                                                )

            sim_graph = (sim_edge_src, sim_edge_tgt, sim_edge_type)
            
            cur_edge_src.extend(sim_graph[0])
            cur_edge_tgt.extend(sim_graph[1])
            cur_edge_type.extend(sim_graph[2])

            current_graph = (cur_edge_src, cur_edge_tgt, cur_edge_type)
            print('Number of edges in augmented graph with sim and syn_triplets: ', len(current_graph[0]))
            
            if args.dataset == 'atomic' or args.dataset[:10] == 'conceptnet':
                # transfer back perform evaluation on cpu
                print('Transfer model back to:', args.device)
                model.to(args.device)

            print("**************** Finish Updating Graph *******************************")
            print('')

        # - - - - - - - - - - - - - - Training Process - - - - - - - - - - - - - - - - - - - - -
        model.train()
        cur_train_data = train_data[:]

        # graph build for current epoch
        g, node_id, node_norm = utils.sample_sub_graph(args, args.graph_batch_size, current_graph)
        node_id_copy = np.copy(node_id)
        node_dict = {v: k for k, v in dict(enumerate(node_id_copy)).items()}

        # Add inverse edges to training samples
        src, dst = np.concatenate((cur_train_data[:, 0], cur_train_data[:, 2])), \
                   np.concatenate((cur_train_data[:, 2], cur_train_data[:, 0]))
        rel = cur_train_data[:, 1]
        rel = np.concatenate((rel, rel + num_rels))
        cur_train_data = np.stack((src, rel, dst)).transpose()

        # Prepare sub-graph training sample
        graph_e1_keys = {}
        for triplet in cur_train_data:
            head = triplet[0]
            rel = triplet[1]
            tail = triplet[2]

            if head in node_id_copy and tail in node_id_copy:
                subgraph_src_idx = node_dict[head] # index in subgraph
                subgraph_tgt_idx = node_dict[tail]
                if (subgraph_src_idx, rel) not in graph_e1_keys:
                    graph_e1_keys[(subgraph_src_idx, rel)] = [subgraph_tgt_idx]
                else:
                    graph_e1_keys[(subgraph_src_idx, rel)].append(subgraph_tgt_idx)
        
        key_list = list(graph_e1_keys.keys())
        random.shuffle(key_list)
        cum_loss = 0.0

        for i in range(0, len(key_list), batch_size):
            
            optimizer.zero_grad()
            
            batch = key_list[i : i + batch_size]

            if len(batch) == 1:
                continue

            e1 = torch.LongTensor([elem[0] for elem in batch])
            rel = torch.LongTensor([elem[1] for elem in batch])

            # e2 -> list of target nodes in subgraph
            e2 = [graph_e1_keys[(k[0], k[1])] for k in batch]
            batch_len = len(batch)

            target = torch.zeros((batch_len, node_id_copy.shape[0]))

            e1 = e1.to(args.device)
            rel = rel.to(args.device)
            target = target.to(args.device)

            # construct target tensor
            for j, inst in enumerate(e2):
                target[j, inst] = 1.0

            # perform label smoothing
            target = ((1.0 - args.label_smoothing_epsilon) * target) + (1.0 / target.size(1)) # Bug: https://github.com/TimDettmers/ConvE/issues/55

            t0 = time.time()

            # Update all embedding matrix / obtain graph embedding
            if i % args.clean_update == 0:
                model.update_whole_embedding_matrix(g, node_id_copy)
            
            # add loss according to degree (Not used)
            #sample_normaliaztion = torch.tensor(1.0/degrees[e1.cpu()]).to(args.device)
            sample_normaliaztion = None
            
            # Compute loss
            loss = model(e1, rel, target, sample_normaliaztion)
            #loss = loss.mean() # for parallel

            cum_loss += loss.cpu().item()

            t1 = time.time()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()

            t2 = time.time()
            forward_time.append(t1 - t0)
            backward_time.append(t2 - t1)

        
        
        t = time.localtime()
        current_time = time.strftime("%D - %H:%M:%S", t)
        print("{} | Epoch {:d} | Loss {:.4f} | Best MRR {:.4f} | Best fMRR {:.4f}| Forward {:.4f}s | Backward {:.4f}s | lr {}".
              format(current_time, epoch, cum_loss, best_mrr, f_best_mrr, forward_time[-1], backward_time[-1], optimizer.param_groups[0]['lr']))
        

        # - - - - - - - - - - - - - - Validation and Testing Process - - - - - - - - - - - - - - - - - - - - -
        if epoch % args.evaluate_every == 0:
            model.eval()
            print("\n")
            print("**************** start eval *******************************")

            eval_edge_src = []
            eval_edge_tgt = []
            eval_edge_type = []

            eval_edge_src.extend(fixed_graph[0])
            eval_edge_tgt.extend(fixed_graph[1])
            eval_edge_type.extend(fixed_graph[2])


            # - - - - - - - - - - - - - - Generate graph for evaluation - - - - - - - - - - - - - - - - - - - - -
            print('Generate graph for evaluation')

            print('First pass for all entity embeddings')
            g_whole, node_id, node_norm = utils.sample_sub_graph(args, 99999999999999, current_graph)

            if args.dataset == 'atomic' or args.dataset[:10] == 'conceptnet':
                print('perform evaluation on cpu')
                # perform evaluation on cpu
                model.cpu()
                g_whole.ndata['id'] = g_whole.ndata['id'].cpu()
                g_whole.ndata['norm'] = g_whole.ndata['norm'].cpu()
                g_whole.edata['type'] = g_whole.edata['type'].cpu()
                valid_data = valid_data.cpu()
                test_data = test_data.cpu()
                filtered_valid = filtered_valid.cpu()
                filtered_test = filtered_test.cpu()

            node_id_copy = np.copy(node_id)

            if model.entity_embedding != None:
                del model.entity_embedding
                model.entity_embedding = None

            torch.cuda.empty_cache()
            model.update_whole_embedding_matrix(g_whole, node_id_copy)


            # create all similarity edges
            if args.dynamic_sim_graph:
                print('Create similarity edges')
                degrees_copy = deepcopy(degrees)
                sim_edge_src, sim_edge_tgt, sim_edge_type = utils.dynamic_graph_gen(
                                                                args, 
                                                                model.entity_embedding.detach().cpu().numpy(), 
                                                                n_ontology=args.n_ontology, 
                                                                inductive_index=[],
                                                                degrees=degrees_copy
                                                                )
                sim_graph = (sim_edge_src, sim_edge_tgt, sim_edge_type)


                eval_edge_src.extend(sim_graph[0])
                eval_edge_tgt.extend(sim_graph[1])
                eval_edge_type.extend(sim_graph[2])
            
            
            eval_graph = (eval_edge_src, eval_edge_tgt, eval_edge_type)
            print('Number of edges evalution graph: ', len(eval_graph[0]))


            # Update whole graph embedding with whole graph
            g_whole, node_id, node_norm = utils.sample_sub_graph(args, 99999999999999, eval_graph)



            # - - - - - - - - - - - - - Update whole graph embedding for evaluation - - - - - - - - - - - - - - -
            # Evluate on CPU
            print('Update whole entity embedding from generated graph')
            if args.dataset == 'atomic' or args.dataset[:10] == 'conceptnet':
                g_whole.ndata['id'] = g_whole.ndata['id'].cpu()
                g_whole.ndata['norm'] = g_whole.ndata['norm'].cpu()
                g_whole.edata['type'] = g_whole.edata['type'].cpu()
            node_id_copy = np.copy(node_id)
            if model.entity_embedding != None:
                del model.entity_embedding
                model.entity_embedding = None
            torch.cuda.empty_cache()

            # Update embedding for whole graph
            model.update_whole_embedding_matrix(g_whole, node_id_copy)

            print('')
            print("===========DEV============")
            mrr_dev = evaluation_utils.ranking_and_hits(args, model, valid_data, all_e1_to_multi_e2, train_network)
            #print("================TEST================")
            #mrr_test = evaluation_utils.ranking_and_hits(args, model, test_data, all_e1_to_multi_e2, train_network)
            print("=========== Filtered DEV============")
            f_mrr_dev = evaluation_utils.ranking_and_hits(args, model, filtered_valid, all_e1_to_multi_e2, train_network)
            #print("================Filtered TEST================")
            #f_mrr_test = evaluation_utils.ranking_and_hits(args, model, filtered_test, all_e1_to_multi_e2, train_network)
            

            # lr scheduler
            scheduler.step(mrr_dev)

            # Save for best MRR
            if mrr_dev < best_mrr:
                if epoch >= args.n_epochs:
                    patient_times += 1
                    if patient_times >= args.patient:
                        print("Early stopping...")
                        break
            else:
                patient_times = 0
                best_mrr = mrr_dev
                print("[saving best model so far]")
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'eval_graph': eval_graph}, args.model_state_file)

            # Save for best filtered MRR
            if f_mrr_dev >= f_best_mrr:
                patient_times = 0
                f_best_mrr = f_mrr_dev
                print("[saving best model so far (filtered mrr)]")
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'eval_graph': eval_graph}, args.model_state_file+'.pt')

            print('Current patient: ', patient_times)


            # Move back to args.device
            if args.dataset == 'atomic' or args.dataset[:10] == 'conceptnet':
                # transfer back 
                print('Transfer model back to:', args.device)
                model.to(args.device)
                g_whole.ndata['id'] = g_whole.ndata['id'].to(args.device)
                g_whole.ndata['norm'] = g_whole.ndata['norm'].to(args.device)
                g_whole.edata['type'] = g_whole.edata['type'].to(args.device)

                valid_data = valid_data.to(args.device)
                test_data = test_data.to(args.device)
                filtered_valid = filtered_valid.to(args.device)
                filtered_test = filtered_test.to(args.device)

            print("**************** end eval *******************************")
            print('')


    # - - - - Training Finished - - - - - - - - - - - - - - -
    # - - - - Testing - 1 - - - - - - - - - - - - - - -

    print("Training done")
    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))

    print("\nStart testing (1)")
    # use best model checkpoint
    checkpoint = torch.load(args.model_state_file)
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    eval_graph = checkpoint['eval_graph']
    print("Using best epoch: {}".format(checkpoint['epoch']))
    # Update whole graph embedding
    g_whole, node_id, node_norm = utils.sample_sub_graph(args, 1000000000, eval_graph)
    if args.dataset == 'atomic' or args.dataset[:10] == 'conceptnet':
        print('perform evaluation on cpu')
        model.cpu()
        g_whole.ndata['id'] = g_whole.ndata['id'].cpu()
        g_whole.ndata['norm'] = g_whole.ndata['norm'].cpu()
        g_whole.edata['type'] = g_whole.edata['type'].cpu()
        valid_data = valid_data.cpu()
        test_data = test_data.cpu()
        filtered_valid = filtered_valid.cpu()
        filtered_test = filtered_test.cpu()

    # update all embedding
    if model.entity_embedding != None:
        del model.entity_embedding
        model.entity_embedding = None
        torch.cuda.empty_cache()
    node_id_copy = np.copy(node_id)
    model.update_whole_embedding_matrix(g_whole, node_id_copy)
    
    print("===========DEV============")
    evaluation_utils.ranking_and_hits(args, model, valid_data, all_e1_to_multi_e2, train_network)
    print("================TEST================")
    evaluation_utils.ranking_and_hits(args, model, test_data, all_e1_to_multi_e2, train_network)




    # - - - - Training Finished - - - - - - - - - - - - - - -
    # - - - - Testing - 2 - - - - - - - - - - - - - - -

    print("\nStart testing (2)")
    # use best model checkpoint
    checkpoint = torch.load(args.model_state_file+ '.pt')
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    eval_graph = checkpoint['eval_graph']
    print("Using best epoch: {}".format(checkpoint['epoch']))
    # Update whole graph embedding
    g_whole, node_id, node_norm = utils.sample_sub_graph(args, 1000000000, eval_graph)
    if args.dataset == 'atomic' or args.dataset[:10] == 'conceptnet':
        print('perform evaluation on cpu')
        model.cpu()
        g_whole.ndata['id'] = g_whole.ndata['id'].cpu()
        g_whole.ndata['norm'] = g_whole.ndata['norm'].cpu()
        g_whole.edata['type'] = g_whole.edata['type'].cpu()
        valid_data = valid_data.cpu()
        test_data = test_data.cpu()
        filtered_valid = filtered_valid.cpu()
        filtered_test = filtered_test.cpu()

    # update all embedding
    if model.entity_embedding != None:
        del model.entity_embedding
        model.entity_embedding = None
        torch.cuda.empty_cache()
    node_id_copy = np.copy(node_id)
    model.update_whole_embedding_matrix(g_whole, node_id_copy)

    print("=========== Filtered DEV============")
    f_mrr_dev = evaluation_utils.ranking_and_hits(args, model, filtered_valid, all_e1_to_multi_e2, train_network)
    print("================Filtered TEST================")
    _ = evaluation_utils.ranking_and_hits(args, model, filtered_test, all_e1_to_multi_e2, train_network)
    







if __name__ == '__main__':

    # Parsing all hyperparameters
    args = parse_args.parse_args()

    # Run main function
    try:
        main(args)
    except KeyboardInterrupt:
        print('Interrupted')





