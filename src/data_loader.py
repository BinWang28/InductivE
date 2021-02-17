from collections import Counter

from reader import AtomicTSVReader, ConceptNetTSVReader
import reader_utils


def load_data(args, train_data_only=False):
    # load graph data
    if args.dataset == "atomic":
        dataset_cls = AtomicTSVReader
        data_dir = "data/atomic/"
    elif args.dataset == "conceptnet-82k":
        dataset_cls = ConceptNetTSVReader
        data_dir = "data/conceptnet-82k/"
    elif args.dataset == "conceptnet-100k":
        dataset_cls = ConceptNetTSVReader
        data_dir = "data/conceptnet-100k/"
    else:
        raise ValueError("Invalid option for dataset.")

    train_network = dataset_cls(args.dataset)
    if not train_data_only:
        dev_network = dataset_cls(args.dataset)
        test_network = dataset_cls(args.dataset)

    train_network.read_network(data_dir=data_dir, split="train")
    train_network.print_summary()
    node_list = train_network.graph.iter_nodes()
    node_degrees = [node.get_degree() for node in node_list]
    degree_counter = Counter(node_degrees)
    avg_degree = sum([k * v for k, v in degree_counter.items()]) / sum([v for k, v in degree_counter.items()])
    print("Average Degree: ", avg_degree)

    # Update nodes to network
    if not train_data_only:
        dev_network.read_network(data_dir=data_dir, split="valid", train_network=train_network)
        test_network.read_network(data_dir=data_dir, split="test", train_network=train_network)

    word_vocab = train_network.graph.node2id

    train_data, _ = reader_utils.prepare_batch_dgl(word_vocab, train_network, train_network)
    if not train_data_only:
        valid_data, _ = reader_utils.prepare_batch_dgl(word_vocab, dev_network, train_network)
        test_data, _ = reader_utils.prepare_batch_dgl(word_vocab, test_network, train_network)

    id2node = {v: k for k, v in word_vocab.items()}

    print('')
    print('Total Nodes: ', len(id2node))
    print('')

    if not train_data_only:
        return train_data, valid_data, test_data, train_network, id2node
    else:
        return train_data, train_network, id2node

