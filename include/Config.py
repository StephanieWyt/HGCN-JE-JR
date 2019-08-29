import tensorflow as tf


class Config:
    language = 'zh_en'  # zh_en | ja_en | fr_en
    e1 = 'data/' + language + '/ent_ids_1'
    e2 = 'data/' + language + '/ent_ids_2'
    r1 = 'data/' + language + '/rel_ids_1'
    r2 = 'data/' + language + '/rel_ids_2'
    ill = 'data/' + language + '/ref_ent_ids'
    ill_r = 'data/' + language + '/ref_r_ids'
    kg1 = 'data/' + language + '/triples_1'
    kg2 = 'data/' + language + '/triples_2'
    epochs = 1000
    dim = 300
    act_func = tf.nn.relu
    gamma = 1.0  # margin based loss
    k = 125  # number of negative samples for each positive one
    seed = 3  # 30% of seeds
    s = 461  # the epoch to switch from preliminary training to joint training, you can try 461 for zh_en, 421 for ja_en, 341 for fr_en 
