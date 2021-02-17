# InductivE

Requirements:
- pytorch=1.4.0
- dgl-cuda10.1
- numpy
- transformers=2.9.1

## Dataset files and pre-computed embeddings

Some files are too large to upload. Please find it through the following link:
https://drive.google.com/drive/folders/1OSKWcv7hmA1oOwcYm4BTKJHX5Zw8OxTF?usp=sharing

## To reproduce the result on ConceptNet-82K

```
    bash train.sh conceptnet-82k 15 saved/saved_ckg_model data/saved_entity_embedding/conceptnet/cn_bert_emb_dict.pkl 500 256 100 ConvTransE 10 1234 1e-25 0.20 0.15 0.15 0.0003 1024 Adam 5 300 RWGCN_NET 50000 1324 data/saved_entity_embedding/conceptnet/cn_fasttext_dict.pkl 300 0.2 5 100 50 0.1
```

## To reproduce the result on ConceptNet-100K

```
    bash train.sh conceptnet-100k 15 saved/saved_ckg_model data/saved_entity_embedding/conceptnet/cn_bert_emb_dict.pkl 500 256 100 ConvTransE 10 1234 1e-20 0.25 0.25 0.25 0.0003 1024 Adam 5 300 RWGCN_NET 50000 1324 data/saved_entity_embedding/conceptnet/cn_fasttext_dict.pkl 300 0.2 5 100 50 0.1
```

## To reproduce the result on ATOMIC

```
    bash train.sh atomic 500 saved/saved_ckg_model data/saved_entity_embedding/atomic/at_bert_emb_dict.pkl 500 256 100 ConvTransE 10 1234 1e-20 0.20 0.20 0.20 0.0001 1024 Adam 5 300 RWGCN_NET 50000 1324 data/saved_entity_embedding/atomic/at_fasttext_dict.pkl 300 0.2 3 100 50 0.1
```



