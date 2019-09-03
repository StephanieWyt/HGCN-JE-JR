# Datasets

> Please first download the datasets [here](http://59.108.48.35/HGCN-JE-JR_data.zip) and extract them into `data/` directory.

Initial datasets are from [GCN-Align](https://github.com/1049451037/GCN-Align) and [JAPE](https://github.com/nju-websoft/JAPE).
We manually aligned more relations from the three datasets and removed the ambiguously aligned relation pairs to construct the test sets for relation alignment.

Take the dataset DBP15K (ZH-EN) as an example, the folder "zh_en" contains:
* ent_ids_1: ids for entities in source KG (ZH);
* ent_ids_2: ids for entities in target KG (EN);
* ref_ent_ids: entity links encoded by ids;
* ref_r_ids: relation links encoded by ids;
* rel_ids_1: ids for entities in source KG (ZH);
* rel_ids_2: ids for entities in target KG (EN);
* triples_1: relation triples encoded by ids in source KG (ZH);
* triples_2: relation triples encoded by ids in target KG (EN);
* zh_vectorList.json: the input entity feature matrix initialized by word vectors;
