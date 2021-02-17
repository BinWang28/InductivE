DATASET=${1}
EVALUATE_EVERY=${2}
OUTPUT_DIR=${3}
BERT_FEAT_PATH=${4}
DECODER_EMBEDDING_DIM=${5}
DECODER_BATCH_SIZE=${6}
N_EPOCHS=${7}
DECODER=${8}
PATIENT=${9}
SEED=${10}
REGULARIZATION=${11}
DROPOUT=${12}
INPUT_DROPOUT=${13}
FEATURE_MAP_DROPOUT=${14}
LR=${15}
BERT_FEAT_DIM=${16}
OPTIMIZER=${17}
DEC_KERNEL_SIZE=${18}
DEC_CHANNELS=${19}
ENCODER=${20}
GRAPH_BATCH_SIZE=${21}
ENTITY_FEAT_DIM=${22}
FASTTEXT_FEAT_PATH=${23}
FASTTEXT_FEAT_DIM=${24}
GNN_DROPOUT=${25}
N_ONTOLOGY=${26}
DYNAMIC_GRAPH_EE_EPOCHS=${27}
START_DYNAMIC_GRAPH=${28}
REL_REGULARIZATION=${29}

echo "Start Training......"


python src/train.py \
--dataset ${DATASET} \
--evaluate_every ${EVALUATE_EVERY} \
--output_dir ${OUTPUT_DIR} \
--bert_feat_path ${BERT_FEAT_PATH} \
--decoder_embedding_dim ${DECODER_EMBEDDING_DIM} \
--decoder_batch_size ${DECODER_BATCH_SIZE} \
--n_epochs ${N_EPOCHS} \
--decoder ${DECODER} \
--patient ${PATIENT} \
--seed ${SEED} \
--regularization ${REGULARIZATION} \
--dropout ${DROPOUT} \
--input_dropout ${INPUT_DROPOUT} \
--feature_map_dropout ${FEATURE_MAP_DROPOUT} \
--lr ${LR} \
--bert_feat_dim ${BERT_FEAT_DIM} \
--optimizer ${OPTIMIZER} \
--dec_kernel_size ${DEC_KERNEL_SIZE} \
--dec_channels ${DEC_CHANNELS} \
--encoder ${ENCODER} \
--graph_batch_size ${GRAPH_BATCH_SIZE} \
--entity_feat_dim ${ENTITY_FEAT_DIM} \
--fasttext_feat_path ${FASTTEXT_FEAT_PATH} \
--fasttext_feat_dim ${FASTTEXT_FEAT_DIM} \
--gnn_dropout ${GNN_DROPOUT} \
--n_ontology ${N_ONTOLOGY} \
--dynamic_graph_ee_epochs ${DYNAMIC_GRAPH_EE_EPOCHS} \
--start_dynamic_graph ${START_DYNAMIC_GRAPH} \
--rel_regularization ${REL_REGULARIZATION} \
