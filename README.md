# HGCN-JE-JR

Source code and datasets for EMNLP 2019 paper: ***[Jointly Learning Entity and Relation Representations for Entity Alignment](https://arxiv.org/pdf/1909.09317.pdf)***.

## Datasets

> Please first download the datasets [here](https://drive.google.com/open?id=1mfaeLXdqFnOHLYBXiTHWI7MLwtfTgPYQ) and extract them into `data/` directory.

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

## Environment

* Python>=3.5
* Tensorflow>=1.8.0
* Scipy
* Numpy

## Running

* Modify language or some other settings in *include/Config.py*
* cd to the directory of *main.py*
* run *main.py*

> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.

> If you have any questions about reproduction, please feel free to email to wyting@pku.edu.cn.

## Citation

If you use this model or code, please cite it as follows:

*Yuting Wu, Xiao Liu, Yansong Feng, Zheng Wang and Dongyan Zhao. Jointly Learning Entity and Relation Representations for Entity Alignment. In: EMNLP 2019.*
