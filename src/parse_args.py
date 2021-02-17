import argparse


def parse_args(args=None):

    parser = argparse.ArgumentParser(description='Options for Commonsense Knowledge Base Completion')

    # General
    parser.add_argument("--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed value")
    parser.add_argument("--output_dir", type=str, required=False, default="saved_models",
                        help="output directory to store metrics and model file")
    parser.add_argument("--no_cuda", action='store_true', default=False,
                        help="prevents using cuda")
    parser.add_argument("--overwrite", action='store_true', default=False,
                        help="overwrite save model")
    parser.add_argument("--load_model", type=str, default=None, help="Path to model file")
    parser.add_argument("--eval_only", action='store_true', default=False,
                        help="only evaluate using an existing model")


    # fixed graph
    parser.add_argument("--fix_triplet_graph", type=bool, default=True,
                        help="Include original triplet graph")
    parser.add_argument("--dynamic_sim_graph", type=bool, default=True,
                        help="Include dynamic similarity graph")

    # dynamic graph
    parser.add_argument("--start_dynamic_graph", type=int, default=250,
                        help="epoch to start dynamic graph")
    parser.add_argument("--dynamic_graph_ee_epochs", type=int, default=10,
                        help="period for dynamic graph update")

    parser.add_argument("--n_ontology", type=float, default=5,
                        help="K-NN Graph")



    # Encoder
    parser.add_argument("--encoder", type=str, default='RWGCN_NET', help="encoder used to compute embedding")
    parser.add_argument("--num_hidden", type=int, default=2, help="number of hidden layers of GNN, excluding the first layer")
    parser.add_argument("--clean_update", type=int, default=1, help="iterations to update the whole matrix")
    parser.add_argument("--entity_feat_dim", type=int, default=1324,
                        help="embedding dimension of all feature")

    parser.add_argument("--bert_feat_path", type=str, required=False, default='None', help="Path to obtained node embedding")
    parser.add_argument("--bert_feat_dim", type=int, default=1024,
                        help="embedding dimension of bert feature")

    parser.add_argument("--fasttext_feat_path", type=str, required=False, default='None', help="Path to obtained node embedding")
    parser.add_argument("--fasttext_feat_dim", type=int, default=300,
                        help="embedding dimension of fasttext feature")

    parser.add_argument("--gnn_dropout", type=float, default=0.2,
                        help="feature map dropout")
    parser.add_argument("--l_relu_ratio", type=float, default=0.2,
                        help="ratio for leaky relu")


    # Decoder
    parser.add_argument("--decoder", type=str, default='ConvTransE', help="decoder used to compute scores")
    parser.add_argument("--decoder_embedding_dim", type=int, default=256,
                        help="embedding dimension of decoder")
    parser.add_argument("--dec_kernel_size", type=int, default=5,
                        help="decoder kernel size")
    parser.add_argument("--dec_channels", type=int, default=200,
                        help="decoder channels")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--input_dropout", type=float, default=0.2,
                        help="input dropout")
    parser.add_argument("--feature_map_dropout", type=float, default=0.2,
                        help="feature map dropout")

    # Training
    parser.add_argument("--graph_batch_size", type=int, default=50000,
                        help="sample sub-graph size based on edges")
    parser.add_argument("--decoder_batch_size", type=int, default=256,
                        help="batch size for decoder")

    parser.add_argument("--optimizer", type=str, default='Adam',
                        help="what optimizer to use")
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="number of minimum training epochs")
    parser.add_argument("--evaluate_every", type=int, default=10,
                        help="perform evaluation every n epochs")
    parser.add_argument("--patient", type=int, default=10,
                        help="Patient times for early stopping")
    parser.add_argument("--regularization", type=float, default=1e-20,
                        help="regularization weight")
    parser.add_argument("--rel_regularization", type=float, default=0.1,
                        help="regularization weight on relation embedding")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--label_smoothing_epsilon", type=float, default=0.1,
                        help="epsilon for performing label smoothing")
    parser.add_argument("--grad_norm", type=float, default=1.0,
                        help="norm to clip gradient to")




    args = parser.parse_args()
    #print(args)

    # Print args one line at a time
    for x in vars(args):
        print(x, ':', vars(args)[x])

    return args


