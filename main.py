import tensorflow as tf
from include.Config import Config
from include.Model import build, training
from include.Test import *
from include.Load import *

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

'''
Followed the code style of GCN-Align:
https://github.com/1049451037/GCN-Align
'''

if __name__ == '__main__':
    e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))

    ILL = loadfile(Config.ill, 2)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * Config.seed])
    test = ILL[illL // 10 * Config.seed:]
    test_r = loadfile(Config.ill_r, 2)
    
    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)
    
    output_prel_e, output_joint_e, output_r, loss_1, loss_2, head, tail = build(Config.dim, Config.act_func,
                                                                                  Config.gamma, Config.k, Config.language[0:2], e, train, KG1 + KG2)
    J = training(output_prel_e, output_joint_e, output_r, loss_1, loss_2, 0.001, Config.epochs, train, e, 
                 Config.k, Config.s, test, test_r, head, tail)
    print('loss:', J)
