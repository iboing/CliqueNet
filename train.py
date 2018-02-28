import os
import argparse
import numpy as np
import tensorflow as tf
import time
from dataloader.data_generator import load_data
from models.cliquenet import build_model

def into_batch(data, label, batch_size, shuffle):
    if shuffle:
        rand_indexes = np.random.permutation(data.shape[0])
        data = data[rand_indexes]
        label = label[rand_indexes]
        
    batch_count=len(data)/batch_size
    batches_data = np.split(data[:batch_count*batch_size], batch_count)
    batches_data.append(data[batch_count*batch_size:])
    batches_labels = np.split(label[:batch_count * batch_size], batch_count)
    batches_labels.append(label[batch_count*batch_size:])
    batch_count+=1
    
    return batches_data, batches_labels, batch_count
        
    
def count_params():
    total_params=0
    for variable in tf.trainable_variables():
        shape=variable.get_shape()
        params=1
        for dim in shape:
            params=params*dim.value
        total_params+=params
    print("Total training params: %.2fM" % (total_params / 1e6))


if __name__=='__main__':
    ##  
    train_params={'normalize_type': 'by-channels',   ## by-channels, divide-255, divide-256
            'initial_lr': 0.1,
            'weight_decay': 1e-4,
            'batch_size': 64,
            'total_epoch': 300,
            'keep_prob':0.8
            }
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default="0")
    parser.add_argument('--dataset',
                        choices=['cifar-10', 'cifar-100', 'svhn'])
    parser.add_argument('--k', type=int, 
                        help='filters per layer')
    parser.add_argument('--T', type=int, 
                        help='total layers in all blocks')
    parser.add_argument('--dir', 
                        help='folder to store models')
    parser.add_argument('--if_a', default=False, type=bool, 
                        help='if use attentional transition')
    parser.add_argument('--if_b', default=False, type=bool, 
                        help='if use bottleneck architecture')
    parser.add_argument('--if_c', default=False, type=bool, 
                        help='if use compression')
        
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    dataset=args.dataset
    
    if dataset=='svhn':
        total_epoches=40
    else:
        total_epoches = train_params['total_epoch']    
        
    result_dir=args.dir
    
    batch_size = train_params['batch_size']
    lr = train_params['initial_lr']
    kp = train_params['keep_prob']
    weight_decay = train_params['weight_decay']
    
    if os.path.exists(result_dir)==False:
        os.mkdir(result_dir)
          
    train_data, train_label, test_data, test_label=load_data(dataset, train_params['normalize_type'])

    image_size=train_data.shape[1:]
    label_num=train_label.shape[-1]

    graph=tf.Graph()
    with graph.as_default():
        input_images=tf.placeholder(tf.float32, [None, image_size[0], image_size[1], image_size[2]], name='input_images')    
        true_labels=tf.placeholder(tf.float32, [None, label_num], name='labels')
        is_train=tf.placeholder(tf.bool, shape=[])
        learning_rate=tf.placeholder(tf.float32, shape=[], name='learning_rate')
        keep_prob=tf.placeholder(tf.float32, shape=[], name='keep_prob')
###     build model
        logits, prob=build_model(input_images, args.k, args.T, label_num, is_train, keep_prob, args.if_a, args.if_b, args.if_c)
###     loss and accuracy  
        loss_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=true_labels))
        if_correct = tf.equal(tf.argmax(prob, 1), tf.argmax(true_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(if_correct, tf.float32))
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
###     optimizer
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        train_op = optimizer.minimize(loss_cross_entropy + l2_loss*weight_decay)
        saver=tf.train.Saver()
        
###     begin training     ###    

    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config, graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        count_params()
        ###   train batch data   ###

        #  to shuffle before each epoch

        ###   test batch data   ###
        
        batches_data_test, batches_labels_test, batch_count_test = into_batch(test_data, test_label, batch_size, shuffle=False)
                
        loss_train=[]
        acc_train=[]
        loss_test=[]
        acc_test=[]
        best_acc=0
        for epoch in range(1, total_epoches+1):
            if epoch == total_epoches/2 : lr=lr*0.1
            if epoch == total_epoches*3/4 : lr=lr*0.1
            
            batches_data, batches_labels, batch_count=into_batch(train_data, train_label, batch_size, shuffle=True)
            
            ###     train     ###
            loss_per_bat=[]
            acc_per_bat=[]
            for batch_id in range(batch_count):
                data_per_bat = batches_data[batch_id]
                label_per_bat = batches_labels[batch_id]
                result_per_bat = sess.run([train_op, loss_cross_entropy, accuracy],
                                feed_dict={input_images : data_per_bat,
                                           true_labels : label_per_bat,
                                           learning_rate : lr,
                                           is_train : True,
                                           keep_prob: kp})
                loss_per_bat.append(result_per_bat[1])
                acc_per_bat.append(result_per_bat[2])
                if (batch_id+1) % 100==0:
                    print 'epoch:', epoch, 'batch:', batch_id+1, 'in', batch_count
                    print 'loss:', result_per_bat[1], 'accuracy:', result_per_bat[2]
                
            saver.save(sess, os.path.join(result_dir, dataset+'_epoch_%d.ckpt' % epoch))
            loss_train.append(np.mean(loss_per_bat))
            acc_train.append(np.mean(acc_per_bat))
            
            ###     test     ###
            loss_per_bat=[]
            acc_per_bat=[]
            for batch_id in range(batch_count_test):
                data_per_bat = batches_data_test[batch_id]
                label_per_bat = batches_labels_test[batch_id]                
                result_per_bat = sess.run([loss_cross_entropy, accuracy],
                                feed_dict={input_images : data_per_bat,
                                           true_labels : label_per_bat,
                                           is_train: False,
                                           keep_prob: 1})
                loss_per_bat.append(result_per_bat[0])     ## result[0]->loss
                acc_per_bat.append(result_per_bat[1])      ## result[1]->acc 
            loss_test.append(np.mean(loss_per_bat))
            acc_test.append(np.mean(acc_per_bat))
            
            if acc_test[-1]>best_acc:
                best_acc=acc_test[-1]      
            
            print time.ctime()
            print 'epoch:',epoch
            print 'train loss:', loss_train[-1],'acc:',acc_train[-1]
            print 'test loss:', loss_test[-1], 'acc:', acc_test[-1]
            print 'best test acc:', best_acc
            print '\n'
            
        np.save(os.path.join(result_dir, result_dir+'_loss_train.npy'), np.array(loss_train))
        np.save(os.path.join(result_dir, result_dir+'_acc_train.npy'), np.array(acc_train))        
        np.save(os.path.join(result_dir, result_dir+'_loss_test.npy'), np.array(loss_test))
        np.save(os.path.join(result_dir, result_dir+'_acc_test.npy'), np.array(acc_test))         
    