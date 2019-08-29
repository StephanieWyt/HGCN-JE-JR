import math
from .Init import *
from include.Test import *
import scipy
import json

def rfunc(KG, e):
    head = {}
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
            tail[tri[1]].add(tri[2])
    r_num = len(head)
    head_r = np.zeros((e, r_num))
    tail_r = np.zeros((e, r_num))
    for tri in KG:
        head_r[tri[0]][tri[1]] = 1
        tail_r[tri[2]][tri[1]] = 1

    return head, tail, head_r, tail_r


def get_mat(e, KG):
    du = [1] * e
    for tri in KG:
        if tri[0] != tri[2]:
            du[tri[0]] += 1
            du[tri[2]] += 1
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 1
        else:
            pass
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = 1
        else:
            pass

    for i in range(e):
        M[(i, i)] = 1
    return M, du


# get a sparse tensor based on relational triples
def get_sparse_tensor(e, KG):
    print('getting a sparse tensor...')
    M, du = get_mat(e, KG)
    ind = []
    val = []
    for fir, sec in M:
        ind.append((sec, fir))
        val.append(M[(fir, sec)] / math.sqrt(du[fir]) / math.sqrt(du[sec]))
    M = tf.SparseTensor(indices=ind, values=val, dense_shape=[e, e])
    
    return M


def add_diag_layer(inlayer, dimension, M, act_func, dropout=0.0, init=ones):
    inlayer = tf.nn.dropout(inlayer, 1 - dropout)
    print('adding a diag layer...')
    w0 = init([1, dimension])
    tosum = tf.sparse_tensor_dense_matmul(M, tf.multiply(inlayer, w0))
    if act_func is None:
        return tosum
    else:
        return act_func(tosum)


def add_full_layer(inlayer, dimension_in, dimension_out, M, act_func, dropout=0.0, init=glorot):
    inlayer = tf.nn.dropout(inlayer, 1 - dropout)
    print('adding a full layer...')
    w0 = init([dimension_in, dimension_out])
    tosum = tf.sparse_tensor_dense_matmul(M, tf.matmul(inlayer, w0))
    if act_func is None:
        return tosum
    else:
        return act_func(tosum)


def highway(layer1,layer2,dimension):
    kernel_gate = glorot([dimension,dimension])
    bias_gate = zeros([dimension])
    transform_gate = tf.matmul(layer1, kernel_gate) + bias_gate
    transform_gate = tf.nn.sigmoid(transform_gate)
    carry_gate = 1.0 - transform_gate
    return transform_gate * layer2 + carry_gate * layer1


def compute_r(inlayer,head_r,tail_r,dimension):
    head_l=tf.transpose(tf.constant(head_r,dtype=tf.float32))
    tail_l=tf.transpose(tf.constant(tail_r,dtype=tf.float32))
    L=tf.matmul(head_l,inlayer)/tf.expand_dims(tf.reduce_sum(head_l,axis=-1),-1)
    R=tf.matmul(tail_l,inlayer)/tf.expand_dims(tf.reduce_sum(tail_l,axis=-1),-1)
    r_embeddings=tf.concat([L,R],axis=-1)
    w_r = glorot([600, 100])
    r_embeddings_new = tf.matmul(r_embeddings, w_r)
    return r_embeddings_new


def compute_joint_e(inlayer,r_embeddings,head_r,tail_r):
    head_r=tf.constant(head_r,dtype=tf.float32)
    tail_r=tf.constant(tail_r,dtype=tf.float32)
    L=tf.matmul(head_r,r_embeddings)
    R=tf.matmul(tail_r,r_embeddings)
    ent_embeddings_new=tf.concat([inlayer, L+R],axis=-1)
    return ent_embeddings_new


def get_input_layer(e, dimension, lang):
    print('adding the primal input layer...')
    with open(file='data/' + lang + '_en/' + lang + '_vectorList.json', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
    input_embeddings = tf.convert_to_tensor(embedding_list)
    ent_embeddings = tf.Variable(input_embeddings)
    return tf.nn.l2_normalize(ent_embeddings, 1)


def get_loss(outlayer, ILL, gamma, k, neg_left, neg_right, neg2_left, neg2_right):
    print('getting loss...')
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    D = A + gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg2_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg2_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
    return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)


def build(dimension, act_func, gamma, k, lang, e, ILL, KG):
    tf.reset_default_graph()
    input_layer = get_input_layer(e, dimension, lang)
    M = get_sparse_tensor(e, KG)
    head, tail, head_r, tail_r = rfunc(KG, e)
    
    print('calculate preliminary entity representations')
    gcn_layer_1 = add_diag_layer(input_layer, dimension, M, act_func, dropout=0.0)
    gcn_layer_1 = highway(input_layer,gcn_layer_1,dimension)
    gcn_layer_2 = add_diag_layer(gcn_layer_1, dimension, M, act_func, dropout=0.0)
    output_prel_e = highway(gcn_layer_1,gcn_layer_2,dimension)
    print('calculate relation representations')
    output_r = compute_r(output_prel_e, head_r, tail_r, dimension)
    print('calculate joint entity representations')
    output_joint_e = compute_joint_e(output_prel_e, output_r, head_r, tail_r)
    
    t = len(ILL)
    neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
    neg2_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg2_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
    loss_1 = get_loss(output_prel_e, ILL, gamma, k, neg_left, neg_right, neg2_left, neg2_right)
    loss_2 = get_loss(output_joint_e, ILL, gamma, k, neg_left, neg_right, neg2_left, neg2_right)
    
    return output_prel_e, output_joint_e, output_r, loss_1, loss_2, head, tail


# get negative samples
def get_neg(ILL, output_layer, k):
    neg = []
    t = len(ILL)
    ILL_vec = np.array([output_layer[e1] for e1 in ILL])
    KG_vec = np.array(output_layer)
    sim = scipy.spatial.distance.cdist(ILL_vec, KG_vec, metric='cityblock')
    for i in range(t):
        rank = sim[i, :].argsort()
        neg.append(rank[0:k])

    neg = np.array(neg)
    neg = neg.reshape((t * k,))
    return neg


def training(output_prel_e, output_joint_e, output_r, loss_1, loss_2, learning_rate, epochs, ILL, e, k, s, test, test_r, head, tail):
    train_step_1 = tf.train.AdamOptimizer(learning_rate).minimize(loss_1)
    train_step_2 = tf.train.AdamOptimizer(learning_rate).minimize(loss_2)
    print('initializing...')
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print('running...')
    J = []
    t = len(ILL)
    ILL = np.array(ILL)
    L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
    neg_left = L.reshape((t * k,))
    L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
    neg2_right = L.reshape((t * k,))
    print('detect coincidence')
    coinc=detect_coinc(test_r, head, tail, ILL)

    for i in range(epochs):
        if i<s:  # preliminary training
            if i % 50 == 0:
                out = sess.run(output_prel_e)
                neg2_left = get_neg(ILL[:, 1], out, k)
                neg_right = get_neg(ILL[:, 0], out, k)
                feeddict = {"neg_left:0": neg_left,
                            "neg_right:0": neg_right,
                            "neg2_left:0": neg2_left,
                            "neg2_right:0": neg2_right}

            sess.run(train_step_1, feed_dict=feeddict)
            if i % 10 == 0:
                th, outvec_e, outvec_r = sess.run([loss_1, output_prel_e, output_r],
                                                feed_dict=feeddict)
                J.append(th)
                get_hits(outvec_e, test)
                if i == s-1 or i==0:
                    get_hits_rel(outvec_r, test_r, coinc)

        else:  # joint training
            if i % 50 == 0:
                out = sess.run(output_joint_e)
                neg2_left = get_neg(ILL[:, 1], out, k)
                neg_right = get_neg(ILL[:, 0], out, k)
                feeddict = {"neg_left:0": neg_left,
                            "neg_right:0": neg_right,
                            "neg2_left:0": neg2_left,
                            "neg2_right:0": neg2_right}

            sess.run(train_step_2, feed_dict=feeddict)
            if i % 10 == 0:
                th, outvec_e, outvec_r = sess.run([loss_2, output_joint_e, output_r],
                                                feed_dict=feeddict)
                J.append(th)
                get_hits(outvec_e, test)
                get_hits_rel(outvec_r, test_r, coinc)
        print('%d/%d' % (i + 1, epochs), 'epochs...')

    sess.close()
    return J
