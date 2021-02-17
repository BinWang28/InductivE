from graph import Graph

import csv
import json
import os
import pandas as pd
import random
import numpy as np


class Reader:

    def print_summary(self):

        print("\n\nGraph Summary")
        print("\nNodes: %d" % len(self.graph.nodes))
        print("Edges: %d" % self.graph.edgeCount)
        print("Relations: %d" % len(self.graph.relation2id))
        density = self.graph.edgeCount / (len(self.graph.nodes) * (len(self.graph.nodes) - 1))
        print("Density: %f" % density)

        print("\n******************* Sample Edges *******************")

        for i, edge in enumerate(self.graph.iter_edges()):
            print(edge)
            if (i+1) % 10 == 0:
                break

        print("***************** ***************** *****************\n")



class ConceptNetTSVReader(Reader):

    def __init__(self, dataset):
        print("Reading ConceptNet")
        self.dataset = dataset
        self.graph = Graph()
        self.rel2id = {}

    def read_network(self, data_dir, split="train", train_network=None):

        if split == "train":
            data_path = os.path.join(data_dir, "train.txt")
            #data_path = os.path.join(data_dir, "train_old.txt")
        elif split == "valid":
            data_path = os.path.join(data_dir, "valid.txt")
            #data_path = os.path.join(data_dir, "valid_old.txt")
        elif split == "test":
            data_path = os.path.join(data_dir, "test.txt")
            #data_path = os.path.join(data_dir, "test_old.txt")

        with open(data_path) as f:
            data = f.readlines()

        acc_add_nodes = 0
        for inst in data:
            inst = inst.strip()
            if inst:
                inst = inst.split('\t')
                rel, src, tgt = inst
                weight = 1.0
                src = src.lower()
                tgt = tgt.lower()
                if split != "train":
                    _, new_added = self.add_example(src, tgt, rel, float(weight), int(weight), train_network)
                    acc_add_nodes += new_added
                else:
                    self.add_example(src, tgt, rel, float(weight))

        print('Number of OOV nodes in {}: {}'.format(split, acc_add_nodes))

        self.rel2id = self.graph.relation2id

    def add_example(self, src, tgt, relation, weight, label=1, train_network=None):

        src_id = self.graph.find_node(src)
        if src_id == -1:
            src_id = self.graph.add_node(src)

        tgt_id = self.graph.find_node(tgt)
        if tgt_id == -1:
            tgt_id = self.graph.add_node(tgt)

        relation_id = self.graph.find_relation(relation)
        if relation_id == -1:
            relation_id = self.graph.add_relation(relation)

        edge = self.graph.add_edge(self.graph.nodes[src_id],
                                   self.graph.nodes[tgt_id],
                                   self.graph.relations[relation_id],
                                   label,
                                   weight)

        # add nodes/relations from evaluation graphs to training graph too
        new_added = 0
        if train_network is not None and label == 1:
            src_id = train_network.graph.find_node(src)
            if src_id == -1:
                src_id = train_network.graph.add_node(src)
                new_added += 1

            tgt_id = train_network.graph.find_node(tgt)
            if tgt_id == -1:
                tgt_id = train_network.graph.add_node(tgt)
                new_added += 1

            relation_id = train_network.graph.find_relation(relation)
            if relation_id == -1:
                relation_id = train_network.graph.add_relation(relation)

        return edge, new_added



class AtomicTSVReader(Reader):

    def __init__(self, dataset):
        print("Reading ATOMIC corpus in TSV format")

        self.dataset = dataset
        self.graph = Graph()
        self.rel2id = {}

    def read_network(self, data_dir, split="train", train_network=None):

        data_path = data_dir
        filename = split + ".preprocessed.txt"

        with open(os.path.join(data_path, filename)) as f:
            data = f.readlines()

        acc_add_nodes = 0
        for inst in data:
            inst = inst.strip()
            if inst:
                inst = inst.split('\t')
                if len(inst) == 3:
                    src, rel, tgt = inst
                    if split != "train":
                        _, new_added = self.add_example(src, tgt, rel, train_network=train_network)
                        acc_add_nodes += new_added
                    else:
                        self.add_example(src, tgt, rel)

        
        print('Number of OOV nodes in {}: {}'.format(split, acc_add_nodes))


    def add_example(self, src, tgt, relation, weight=1.0, label=1, train_network=None):

        src_id = self.graph.find_node(src)
        if src_id == -1:
            src_id = self.graph.add_node(src)

        tgt_id = self.graph.find_node(tgt)
        if tgt_id == -1:
            tgt_id = self.graph.add_node(tgt)

        relation_id = self.graph.find_relation(relation)
        if relation_id == -1:
            relation_id = self.graph.add_relation(relation)

        edge = self.graph.add_edge(self.graph.nodes[src_id],
                                   self.graph.nodes[tgt_id],
                                   self.graph.relations[relation_id],
                                   label,
                                   weight)

        # add nodes/relations from evaluation graphs to training graph too
        new_added = 0
        if train_network is not None and label == 1:
            src_id = train_network.graph.find_node(src)
            if src_id == -1:
                src_id = train_network.graph.add_node(src)
                new_added += 1

            tgt_id = train_network.graph.find_node(tgt)
            if tgt_id == -1:
                tgt_id = train_network.graph.add_node(tgt)
                new_added += 1

            relation_id = train_network.graph.find_relation(relation)
            if relation_id == -1:
                relation_id = train_network.graph.add_relation(relation)

        return edge, new_added
