import tensorflow as tf
from utils import *

block_num=3

def build_model(input_images, k, T, label_num, is_train, keep_prob, if_a, if_b, if_c):

    current=first_transit(input_images, channels=64, strides=1, with_biase=False)
    current_list=[]

    ## build blocks    
    for i in range(block_num):
        block_feature, transit_feature = loop_block(current, if_b, channels_per_layer=k, layer_num=T/3, is_train=is_train, keep_prob=keep_prob, block_name='b'+str(i))
        if if_c==True:
            block_feature=compress(block_feature, is_train=is_train, keep_prob=keep_prob, name='com'+str(i))
        current_list.append(global_pool(block_feature, is_train))
        if i==block_num-1:
            break
        current=transition(transit_feature, if_a, is_train=is_train, keep_prob=keep_prob, name='tran'+str(i))
        
    
    ## final feature
    final_feature=current_list[0]
    for block_id in range(len(current_list)-1):
        final_feature=tf.concat((final_feature, current_list[block_id+1]),
                                axis=3)
    feature_length=final_feature.get_shape().as_list()[-1]
    print 'final feature length:',feature_length

    feature_flatten=tf.reshape(final_feature, [-1, feature_length])
    ##   final_fc
    Wfc=tf.get_variable(name='FC_W', shape=[feature_length, label_num], initializer=tf.contrib.layers.xavier_initializer())
    bfc=tf.get_variable(name='FC_b', initializer=tf.constant(0.0, shape=[label_num]))

    logits=tf.matmul(feature_flatten, Wfc)+bfc
    prob=tf.nn.softmax(logits)
    
    return logits, prob