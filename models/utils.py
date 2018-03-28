import tensorflow as tf


def bias_var(out_channels, init_method):
    initial_value=tf.constant(0.0, shape=[out_channels])
    biases=tf.Variable(initial_value)
    
    return biases

def conv_var(kernel_size, in_channels, out_channels, init_method, name):
    shape=[kernel_size[0], kernel_size[1], in_channels, out_channels]
    if init_method=='msra':
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer())
    elif init_method=='xavier':
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def attentional_transition(input_layer, name):
    
    channels=input_layer.get_shape().as_list()[-1]
    map_size=input_layer.get_shape().as_list()[1]
 
    bottom_fc=tf.nn.avg_pool(input_layer, [1, map_size, map_size, 1], [1, map_size, map_size, 1], 'VALID')

    assert bottom_fc.get_shape().as_list()[-1]==channels   ## none,1,1,C
    
    bottom_fc=tf.reshape(bottom_fc, [-1, channels])   ## none, C
    
    Wfc=tf.get_variable(name=name+'_W1', shape=[channels, channels/2], initializer=tf.contrib.layers.xavier_initializer())
    bfc=tf.get_variable(name=name+'_b1', initializer=tf.constant(0.0, shape=[channels/2])) 
    
    mid_fc=tf.nn.relu(tf.matmul(bottom_fc, Wfc)+bfc)
    
    Wfc=tf.get_variable(name=name+'_W2', shape=[channels/2, channels], initializer=tf.contrib.layers.xavier_initializer())
    bfc=tf.get_variable(name=name+'_b2', initializer=tf.constant(0.0, shape=[channels]))    
    
    top_fc=tf.nn.sigmoid(tf.matmul(mid_fc, Wfc)+bfc)   ## none, C

    top_fc = tf.reshape(top_fc, [-1, 1, 1, channels])

    output_layer = tf.multiply(input_layer, top_fc)

    return output_layer

def transition(input_layer, if_a, is_train, keep_prob, name):
    channels=input_layer.get_shape().as_list()[-1]
    output_layer=tf.contrib.layers.batch_norm(input_layer, scale=True, is_training=is_train, updates_collections=None)
    output_layer=tf.nn.relu(output_layer)
    filters=conv_var(kernel_size=(1,1), in_channels=channels, out_channels=channels, init_method='msra', name=name)
    output_layer=tf.nn.conv2d(output_layer, filters, [1, 1, 1, 1], padding='SAME')
    output_layer=tf.nn.dropout(output_layer, keep_prob)
    ## attentional transition
    if if_a:
        output_layer=attentional_transition(output_layer, name=name+'-ATT')
    
    output_layer=tf.nn.avg_pool(output_layer, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
    
    return output_layer

def compress(input_layer, is_train, keep_prob, name):
    channels=input_layer.get_shape().as_list()[-1]
    output_layer=tf.contrib.layers.batch_norm(input_layer, scale=True, is_training=is_train, updates_collections=None)
    output_layer=tf.nn.relu(output_layer)
    filters=conv_var(kernel_size=(1,1), in_channels=channels, out_channels=channels/2, init_method='msra', name=name)
    output_layer=tf.nn.conv2d(output_layer, filters, [1, 1, 1, 1], padding='SAME')
    output_layer=tf.nn.dropout(output_layer, keep_prob)
    
    return output_layer


def global_pool(input_layer, is_train):
    output_layer=tf.contrib.layers.batch_norm(input_layer, scale=True, is_training=is_train, updates_collections=None)
    output_layer=tf.nn.relu(output_layer)
    
    map_size=input_layer.get_shape().as_list()[1]
    return tf.nn.avg_pool(output_layer, [1, map_size, map_size, 1], [1, map_size, map_size, 1], 'VALID')

def first_transit(input_layer, channels, strides, with_biase=False):
    filters=conv_var(kernel_size=(3,3), in_channels=3, out_channels=channels, init_method='msra', name='first_tran')
    conved=tf.nn.conv2d(input_layer, filters, [1, strides, strides, 1], padding='SAME')
    if with_biase==True:
        biases=bias_var(out_channels=channels)
        biased=tf.nn.bias_add(conved, biases)
        return biased
    return conved


def loop_block(input_layer, if_b, channels_per_layer, layer_num, is_train, keep_prob, block_name, loop_num=1):
    if if_b: layer_num = layer_num/2 ## if bottleneck is used, the T value should be multiplied by 2.
    channels=channels_per_layer
    node_0_channels=input_layer.get_shape().as_list()[-1]
    ## init param
    param_dict={}
    kernel_size=(1, 1) if if_b==True else (3, 3)
    for layer_id in range(1, layer_num):
        add_id=1
        while layer_id+add_id <= layer_num:
            
            ## ->
            filters=conv_var(kernel_size=kernel_size, in_channels=channels, out_channels=channels, init_method='msra', name=block_name+'-'+str(layer_id)+'_'+str(layer_id+add_id))
            param_dict[str(layer_id)+'_'+str(layer_id+add_id)]=filters
            ## <-
            filters_inv=conv_var(kernel_size=kernel_size, in_channels=channels, out_channels=channels, init_method='msra', name=block_name+'-'+str(layer_id+add_id)+'_'+str(layer_id))
            param_dict[str(layer_id+add_id)+'_'+str(layer_id)]=filters_inv
            add_id+=1
    
    for layer_id in range(layer_num):
        filters=conv_var(kernel_size=kernel_size, in_channels=node_0_channels, out_channels=channels, init_method='msra', name=block_name+'-'+str(0)+'_'+str(layer_id+1))
        param_dict[str(0)+'_'+str(layer_id+1)]=filters
    
    assert len(param_dict)==layer_num*(layer_num-1)+layer_num

    ###   bottleneck param  ###
    if if_b==True:
        param_dict_B={}
        for layer_id in range(1, layer_num+1):
            filters=conv_var(kernel_size=(3,3), in_channels=channels, out_channels=channels, init_method='msra', name=block_name+'-'+'to-'+str(layer_id))
            param_dict_B[str(layer_id)]=filters

    ## init blob
    blob_dict={}

    for layer_id in range(1, layer_num+1):
        bottom_blob=input_layer
        bottom_param=param_dict['0_'+str(layer_id)]
        for layer_id_id in range(1, layer_id):
            bottom_blob=tf.concat((bottom_blob, blob_dict[str(layer_id_id)]), axis=3)
            bottom_param=tf.concat((bottom_param, param_dict[str(layer_id_id)+'_'+str(layer_id)]), axis=2)


        mid_layer=tf.contrib.layers.batch_norm(bottom_blob, scale=True, is_training=is_train, updates_collections=None)
        mid_layer=tf.nn.relu(mid_layer)
        mid_layer=tf.nn.conv2d(mid_layer, bottom_param, [1,1,1,1], padding='SAME')
        mid_layer=tf.nn.dropout(mid_layer, keep_prob)
        ##  Bottle neck
        if if_b==True:
            next_layer=tf.contrib.layers.batch_norm(mid_layer, scale=True, is_training=is_train, updates_collections=None)
            next_layer=tf.nn.relu(next_layer)
            next_layer=tf.nn.conv2d(next_layer, param_dict_B[str(layer_id)], [1,1,1,1], padding='SAME')
            next_layer=tf.nn.dropout(next_layer, keep_prob)
        else:
            next_layer=mid_layer
        blob_dict[str(layer_id)]=next_layer
    
    ## begin loop
    for loop_id in range(loop_num):
        for layer_id in range(1, layer_num+1):    ##   [1,2,3,4,5]
            
            layer_list=[str(l_id) for l_id in range(1, layer_num+1)]
            layer_list.remove(str(layer_id))
            
            bottom_blobs=blob_dict[layer_list[0]]
            bottom_param=param_dict[layer_list[0]+'_'+str(layer_id)]
            for bottom_id in range(len(layer_list)-1):
                bottom_blobs=tf.concat((bottom_blobs, blob_dict[layer_list[bottom_id+1]]),
                        axis=3)   ###  concatenate the data blobs
                bottom_param=tf.concat((bottom_param, param_dict[layer_list[bottom_id+1]+'_'+str(layer_id)]),
                        axis=2)   ###  concatenate the parameters                
            
            mid_layer=tf.contrib.layers.batch_norm(bottom_blobs,  scale=True, is_training=is_train, updates_collections=None)
            mid_layer=tf.nn.relu(mid_layer)            
            mid_layer=tf.nn.conv2d(mid_layer, bottom_param, [1,1,1,1], padding='SAME')    ###  update the data blob
            mid_layer=tf.nn.dropout(mid_layer, keep_prob)
            ## Bottle neck
            if if_b==True:
                next_layer=tf.contrib.layers.batch_norm(mid_layer, scale=True, is_training=is_train, updates_collections=None)
                next_layer=tf.nn.relu(next_layer)
                next_layer=tf.nn.conv2d(next_layer, param_dict_B[str(layer_id)], [1,1,1,1], padding='SAME')
                next_layer=tf.nn.dropout(next_layer, keep_prob)
            else:
                next_layer=mid_layer
            blob_dict[str(layer_id)]=next_layer
    
    transit_feature=blob_dict['1']
    for layer_id in range(2, layer_num+1):
        transit_feature=tf.concat((transit_feature, blob_dict[str(layer_id)]), axis=3)    
    
    block_feature=tf.concat((input_layer, transit_feature), axis=3)
    
    return block_feature, transit_feature

def loop_block_I_I(input_layer, if_b, channels_per_layer, layer_num, is_train, keep_prob, block_name):
    if if_b: layer_num = layer_num/2 ## if bottleneck is used, the T value should be multiplied by 2.
    channels=channels_per_layer
    node_0_channels=input_layer.get_shape().as_list()[-1]
    ## init param
    param_dict={}
    kernel_size=(1, 1) if if_b==True else (3, 3)
    for layer_id in range(1, layer_num):
        add_id=1
        while layer_id+add_id <= layer_num:
            
            ## ->
            filters=conv_var(kernel_size=kernel_size, in_channels=channels, out_channels=channels, init_method='msra', name=block_name+'-'+str(layer_id)+'_'+str(layer_id+add_id))
            param_dict[str(layer_id)+'_'+str(layer_id+add_id)]=filters
            ## <-
            filters_inv=conv_var(kernel_size=kernel_size, in_channels=channels, out_channels=channels, init_method='msra', name=block_name+'-'+str(layer_id+add_id)+'_'+str(layer_id))
            param_dict[str(layer_id+add_id)+'_'+str(layer_id)]=filters_inv
            add_id+=1
    
    for layer_id in range(layer_num):
        filters=conv_var(kernel_size=kernel_size, in_channels=node_0_channels, out_channels=channels, init_method='msra', name=block_name+'-'+str(0)+'_'+str(layer_id+1))
        param_dict[str(0)+'_'+str(layer_id+1)]=filters
    
    assert len(param_dict)==layer_num*(layer_num-1)+layer_num

    ###   bottleneck param  ###
    if if_b==True:
        param_dict_B={}
        for layer_id in range(1, layer_num+1):
            filters=conv_var(kernel_size=(3,3), in_channels=channels, out_channels=channels, init_method='msra', name=block_name+'-'+'to-'+str(layer_id))
            param_dict_B[str(layer_id)]=filters

    ## init blob
    blob_dict={}

    for layer_id in range(1, layer_num+1):
        bottom_blob=input_layer
        bottom_param=param_dict['0_'+str(layer_id)]
        for layer_id_id in range(1, layer_id):
            bottom_blob=tf.concat((bottom_blob, blob_dict[str(layer_id_id)]), axis=3)
            bottom_param=tf.concat((bottom_param, param_dict[str(layer_id_id)+'_'+str(layer_id)]), axis=2)


        mid_layer=tf.contrib.layers.batch_norm(bottom_blob, scale=True, is_training=is_train, updates_collections=None)
        mid_layer=tf.nn.relu(mid_layer)
        mid_layer=tf.nn.conv2d(mid_layer, bottom_param, [1,1,1,1], padding='SAME')
        mid_layer=tf.nn.dropout(mid_layer, keep_prob)
        ##  Bottle neck
        if if_b==True:
            next_layer=tf.contrib.layers.batch_norm(mid_layer, scale=True, is_training=is_train, updates_collections=None)
            next_layer=tf.nn.relu(next_layer)
            next_layer=tf.nn.conv2d(next_layer, param_dict_B[str(layer_id)], [1,1,1,1], padding='SAME')
            next_layer=tf.nn.dropout(next_layer, keep_prob)
        else:
            next_layer=mid_layer
        blob_dict[str(layer_id)]=next_layer
    
    ## no loop
    
    transit_feature=blob_dict['1']
    for layer_id in range(2, layer_num+1):
        transit_feature=tf.concat((transit_feature, blob_dict[str(layer_id)]), axis=3)    
    
    block_feature=tf.concat((input_layer, transit_feature), axis=3)
    
    return block_feature, transit_feature    


def loop_block_I_II(input_layer, if_b, channels_per_layer, layer_num, is_train, keep_prob, block_name, loop_num=1):
    if if_b: layer_num = layer_num/2 ## if bottleneck is used, the T value should be multiplied by 2.
    import copy

    channels=channels_per_layer
    node_0_channels=input_layer.get_shape().as_list()[-1]
    ## init param
    param_dict={}
    kernel_size=(1, 1) if if_b==True else (3, 3)
    for layer_id in range(1, layer_num):
        add_id=1
        while layer_id+add_id <= layer_num:
            
            ## ->
            filters=conv_var(kernel_size=kernel_size, in_channels=channels, out_channels=channels, init_method='msra', name=block_name+'-'+str(layer_id)+'_'+str(layer_id+add_id))
            param_dict[str(layer_id)+'_'+str(layer_id+add_id)]=filters
            ## <-
            filters_inv=conv_var(kernel_size=kernel_size, in_channels=channels, out_channels=channels, init_method='msra', name=block_name+'-'+str(layer_id+add_id)+'_'+str(layer_id))
            param_dict[str(layer_id+add_id)+'_'+str(layer_id)]=filters_inv
            add_id+=1
    
    for layer_id in range(layer_num):
        filters=conv_var(kernel_size=kernel_size, in_channels=node_0_channels, out_channels=channels, init_method='msra', name=block_name+'-'+str(0)+'_'+str(layer_id+1))
        param_dict[str(0)+'_'+str(layer_id+1)]=filters
    
    assert len(param_dict)==layer_num*(layer_num-1)+layer_num

    ###   bottleneck param  ###
    if if_b==True:
        param_dict_B={}
        for layer_id in range(1, layer_num+1):
            filters=conv_var(kernel_size=(3,3), in_channels=channels, out_channels=channels, init_method='msra', name=block_name+'-'+'to-'+str(layer_id))
            param_dict_B[str(layer_id)]=filters

    ## init blob
    blob_dict={}
    blob_dict_list=[]

    for layer_id in range(1, layer_num+1):
        bottom_blob=input_layer
        bottom_param=param_dict['0_'+str(layer_id)]
        for layer_id_id in range(1, layer_id):
            bottom_blob=tf.concat((bottom_blob, blob_dict[str(layer_id_id)]), axis=3)
            bottom_param=tf.concat((bottom_param, param_dict[str(layer_id_id)+'_'+str(layer_id)]), axis=2)


        mid_layer=tf.contrib.layers.batch_norm(bottom_blob, scale=True, is_training=is_train, updates_collections=None)
        mid_layer=tf.nn.relu(mid_layer)
        mid_layer=tf.nn.conv2d(mid_layer, bottom_param, [1,1,1,1], padding='SAME')
        mid_layer=tf.nn.dropout(mid_layer, keep_prob)
        ##  Bottle neck
        if if_b==True:
            next_layer=tf.contrib.layers.batch_norm(mid_layer, scale=True, is_training=is_train, updates_collections=None)
            next_layer=tf.nn.relu(next_layer)
            next_layer=tf.nn.conv2d(next_layer, param_dict_B[str(layer_id)], [1,1,1,1], padding='SAME')
            next_layer=tf.nn.dropout(next_layer, keep_prob)
        else:
            next_layer=mid_layer
        blob_dict[str(layer_id)]=next_layer
    
    blob_dict_list.append(blob_dict)
    
    ## begin loop
    for loop_id in range(loop_num):
        blob_dict_new = copy.copy(blob_dict_list[-1])
        for layer_id in range(1, layer_num+1):    ##   [1,2,3,4,5]
            
            layer_list=[str(l_id) for l_id in range(1, layer_num+1)]
            layer_list.remove(str(layer_id))
            
            bottom_blobs=blob_dict_new[layer_list[0]]
            bottom_param=param_dict[layer_list[0]+'_'+str(layer_id)]
            for bottom_id in range(len(layer_list)-1):
                bottom_blobs=tf.concat((bottom_blobs, blob_dict_new[layer_list[bottom_id+1]]),
                        axis=3)   ###  concatenate the data blobs
                bottom_param=tf.concat((bottom_param, param_dict[layer_list[bottom_id+1]+'_'+str(layer_id)]),
                        axis=2)   ###  concatenate the parameters                
            
            mid_layer=tf.contrib.layers.batch_norm(bottom_blobs,  scale=True, is_training=is_train, updates_collections=None)
            mid_layer=tf.nn.relu(mid_layer)            
            mid_layer=tf.nn.conv2d(mid_layer, bottom_param, [1,1,1,1], padding='SAME')    ###  update the data blob
            mid_layer=tf.nn.dropout(mid_layer, keep_prob)
            ## Bottle neck
            if if_b==True:
                next_layer=tf.contrib.layers.batch_norm(mid_layer, scale=True, is_training=is_train, updates_collections=None)
                next_layer=tf.nn.relu(next_layer)
                next_layer=tf.nn.conv2d(next_layer, param_dict_B[str(layer_id)], [1,1,1,1], padding='SAME')
                next_layer=tf.nn.dropout(next_layer, keep_prob)
            else:
                next_layer=mid_layer
            blob_dict_new[str(layer_id)]=next_layer
        blob_dict_list.append(blob_dict_new)
    
    assert len(blob_dict_list)==1+loop_num
    
    stage_I = blob_dict_list[0]['1']
    for layer_id in range(2, layer_num+1):
        stage_I=tf.concat((stage_I, blob_dict_list[0][str(layer_id)]), axis=3)     
              
    stage_II = blob_dict_list[1]['1']
    for layer_id in range(2, layer_num+1):
        stage_II=tf.concat((stage_II, blob_dict_list[1][str(layer_id)]), axis=3)        
                  
    block_feature = tf.concat((input_layer, stage_I), axis=3)
    transit_feature = stage_II
    
    return block_feature, transit_feature    
    

def loop_block_X(input_layer, x_value, if_b, channels_per_layer, layer_num, is_train, keep_prob, block_name, loop_num=1):
    if if_b: layer_num = layer_num/2 ## if bottleneck is used, the T value should be multiplied by 2.
    channels=channels_per_layer
    node_0_channels=input_layer.get_shape().as_list()[-1]
    ## init param
    param_dict={}
    kernel_size=(1, 1) if if_b==True else (3, 3)
    for layer_id in range(1, layer_num):
        add_id=1
        while layer_id+add_id <= layer_num:
            
            ## ->
            filters=conv_var(kernel_size=kernel_size, in_channels=channels, out_channels=channels, init_method='msra', name=block_name+'-'+str(layer_id)+'_'+str(layer_id+add_id))
            param_dict[str(layer_id)+'_'+str(layer_id+add_id)]=filters
            ## <-
            filters_inv=conv_var(kernel_size=kernel_size, in_channels=channels, out_channels=channels, init_method='msra', name=block_name+'-'+str(layer_id+add_id)+'_'+str(layer_id))
            param_dict[str(layer_id+add_id)+'_'+str(layer_id)]=filters_inv
            add_id+=1
    
    for layer_id in range(layer_num):
        filters=conv_var(kernel_size=kernel_size, in_channels=node_0_channels, out_channels=channels, init_method='msra', name=block_name+'-'+str(0)+'_'+str(layer_id+1))
        param_dict[str(0)+'_'+str(layer_id+1)]=filters
    
    assert len(param_dict)==layer_num*(layer_num-1)+layer_num

    ###   bottleneck param  ###
    if if_b==True:
        param_dict_B={}
        for layer_id in range(1, layer_num+1):
            filters=conv_var(kernel_size=(3,3), in_channels=channels, out_channels=channels, init_method='msra', name=block_name+'-'+'to-'+str(layer_id))
            param_dict_B[str(layer_id)]=filters

    ## init blob
    blob_dict={}

    for layer_id in range(1, layer_num+1):
        bottom_blob=input_layer
        bottom_param=param_dict['0_'+str(layer_id)]
        for layer_id_id in range(1, layer_id):
            bottom_blob=tf.concat((bottom_blob, blob_dict[str(layer_id_id)]), axis=3)
            bottom_param=tf.concat((bottom_param, param_dict[str(layer_id_id)+'_'+str(layer_id)]), axis=2)


        mid_layer=tf.contrib.layers.batch_norm(bottom_blob, scale=True, is_training=is_train, updates_collections=None)
        mid_layer=tf.nn.relu(mid_layer)
        mid_layer=tf.nn.conv2d(mid_layer, bottom_param, [1,1,1,1], padding='SAME')
        mid_layer=tf.nn.dropout(mid_layer, keep_prob)
        ##  Bottle neck
        if if_b==True:
            next_layer=tf.contrib.layers.batch_norm(mid_layer, scale=True, is_training=is_train, updates_collections=None)
            next_layer=tf.nn.relu(next_layer)
            next_layer=tf.nn.conv2d(next_layer, param_dict_B[str(layer_id)], [1,1,1,1], padding='SAME')
            next_layer=tf.nn.dropout(next_layer, keep_prob)
        else:
            next_layer=mid_layer
        blob_dict[str(layer_id)]=next_layer
    
    ## begin loop
    for loop_id in range(loop_num):
        for layer_id in range(1, x_value+1):    ##   [1,2,3,4,5]
            
            layer_list=[str(l_id) for l_id in range(1, layer_num+1)]
            layer_list.remove(str(layer_id))
            
            bottom_blobs=blob_dict[layer_list[0]]
            bottom_param=param_dict[layer_list[0]+'_'+str(layer_id)]
            for bottom_id in range(len(layer_list)-1):
                bottom_blobs=tf.concat((bottom_blobs, blob_dict[layer_list[bottom_id+1]]),
                        axis=3)   ###  concatenate the data blobs
                bottom_param=tf.concat((bottom_param, param_dict[layer_list[bottom_id+1]+'_'+str(layer_id)]),
                        axis=2)   ###  concatenate the parameters                
            
            mid_layer=tf.contrib.layers.batch_norm(bottom_blobs,  scale=True, is_training=is_train, updates_collections=None)
            mid_layer=tf.nn.relu(mid_layer)            
            mid_layer=tf.nn.conv2d(mid_layer, bottom_param, [1,1,1,1], padding='SAME')    ###  update the data blob
            mid_layer=tf.nn.dropout(mid_layer, keep_prob)
            ## Bottle neck
            if if_b==True:
                next_layer=tf.contrib.layers.batch_norm(mid_layer, scale=True, is_training=is_train, updates_collections=None)
                next_layer=tf.nn.relu(next_layer)
                next_layer=tf.nn.conv2d(next_layer, param_dict_B[str(layer_id)], [1,1,1,1], padding='SAME')
                next_layer=tf.nn.dropout(next_layer, keep_prob)
            else:
                next_layer=mid_layer
            blob_dict[str(layer_id)]=next_layer
    
    transit_feature=blob_dict['1']
    for layer_id in range(2, layer_num+1):
        transit_feature=tf.concat((transit_feature, blob_dict[str(layer_id)]), axis=3)    
    
    block_feature=tf.concat((input_layer, transit_feature), axis=3)
    
    return block_feature, transit_feature
