from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Input Shape: [-1, 799, 64, 3]
# Layer1 Shape: [-1, 400, 32, 32]
# Layer2 Shape: [-1, 200, 16, 64]
# Layer3 Shape: [-1, 100, 8, 128]
# Layer3 Shape: [-1, 50, 4, 256]

def inference(x, batch_size, keep_probability, phase_train, bottleneck_layer_size, weight_decay):
    # NHWC
    with tf.name_scope('Graph'):
        #x=tf.reshape(x,[batch_size,-1,64,3], name='input')
        print()
        print('-'*100)
        print("INPUT SHAPE: ", x.get_shape().as_list())
        
        with tf.variable_scope('conv2D_A'):
            x = tf.layers.conv2d(x,
                                 filters=32,
                                 kernel_size=5,
                                 strides=1,
                                 padding='SAME',
                                 data_format='channels_last',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),
                                 activation=tf.nn.relu6,
                                 name='conv2D_64_A')
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2,2], padding='SAME')
            
        print("CONVA SHAPE: ", x.get_shape().as_list())
        
        with tf.variable_scope('conv2D_B'):
            x = tf.layers.conv2d(x,
                                 filters=64,
                                 kernel_size=3,
                                 strides=1,
                                 padding='SAME',
                                 data_format = 'channels_last',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),
                                 activation=tf.nn.relu6,
                                 name='conv2D_64_B')
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.layers.max_pooling2d(x, [2, 2], strides=[2,2], padding='SAME')

        print("CONVB SHAPE: ", x.get_shape().as_list())
        
        with tf.variable_scope('Reshape'):
            x = tf.reshape(x, [batch_size, 200*16*64])
            
        print("RESHAPE 2 SHAPE: ", x.get_shape().as_list())
        
        with tf.variable_scope('Fully_connc_1'):
            x = tf.layers.dense(x,
                                units=2048, activation=tf.nn.relu6,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
            
        print("FULLY CONNECTED 1 SHAPE: ", x.get_shape().as_list())
        
        with tf.variable_scope('Fully_connc_2'):
            x = tf.layers.dense(x,
                                units=512,
                                activation=tf.nn.relu6,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
            
        print("FULLY CONNECTED 2 SHAPE: ", x.get_shape().as_list())

        with tf.variable_scope('Affine_Layer_A'):
            x = tf.layers.dense(x, 
                                units=256, 
                                activation = tf.nn.relu6,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))

        print("AFFINE SHAPE: ", x.get_shape().as_list())
        
        print('-'*100)
        print()
        
    return x



def inference_2(x, batch_size, keep_probability, phase_train, bottleneck_layer_size, weight_decay):
    # NHWC
    with tf.name_scope('Graph'):
        #x=tf.reshape(x,[batch_size,-1,64,3], name='input')
        print()
        print('-'*100)
        print("INPUT SHAPE: ", x.get_shape().as_list())
        
        with tf.variable_scope('conv2D_A'):
            x = tf.layers.conv2d(x,
                                 filters=32,
                                 kernel_size=5,
                                 strides=1,
                                 padding='SAME',
                                 data_format='channels_last',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),
                                 activation=tf.nn.relu6,
                                 name='conv2D_64_A')
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2,2], padding='SAME')
            
        print("CONVA SHAPE: ", x.get_shape().as_list())
        
        with tf.variable_scope('conv2D_B'):
            x = tf.layers.conv2d(x,
                                 filters=64,
                                 kernel_size=3,
                                 strides=1,
                                 padding='SAME',
                                 data_format = 'channels_last',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),
                                 activation=tf.nn.relu6,
                                 name='conv2D_64_B')
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.layers.max_pooling2d(x, [2, 2], strides=[2,2], padding='SAME')

        print("CONVB SHAPE: ", x.get_shape().as_list())
        
        x=tf.reshape(x,[batch_size,-1,1024])
        x=tf.transpose(x,[1,0,2])

        print("RESHAPE: ", x.get_shape().as_list())
        
        cells = []
        for _ in range(3):
            cell = tf.contrib.rnn.GRUCell(512)  # Or LSTMCell(num_units)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - keep_probability)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        
        output, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        
        print("GRU SHAPE: ", output.get_shape().as_list())
        
        with tf.variable_scope('Temporal_Average_Layer'):
            x = tf.reduce_mean(output,0)

        print("TEMPORAL AVG SHAPE: ", x.get_shape().as_list())

        with tf.variable_scope('Affine_Layer_A'):
            x = tf.layers.dense(x, 
                                units=256, 
                                activation = tf.nn.relu6,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))

        print("AFFINE SHAPE: ", x.get_shape().as_list())
        
        print('-'*100)
        print()
        
    return x

def inference_val(x, batch_size, keep_probability, phase_train, bottleneck_layer_size, weight_decay):
    # NCHW
    with tf.name_scope('Graph'):
        x=tf.reshape(x,[batch_size,-1,64,3],name='input')
        print()
        print('-'*100)
        print("INPUT SHAPE: ", x.get_shape().as_list())
        
        with tf.variable_scope('conv2D_A'):
            x = tf.layers.conv2d(x,
                                 filters=32,
                                 kernel_size=5,
                                 strides=1,
                                 padding='SAME',
                                 data_format='channels_last',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),
                                 activation=tf.nn.relu6,
                                 name='conv2D_64_A')
            x = tf.layers.batch_normalization(x, training=False)
            x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2,2], padding='SAME')
            
        print("CONVA SHAPE: ", x.get_shape().as_list())
        
        with tf.variable_scope('conv2D_B'):
            x = tf.layers.conv2d(x,
                                 filters=64,
                                 kernel_size=3,
                                 strides=1,
                                 padding='SAME',
                                 data_format = 'channels_last',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),
                                 activation=tf.nn.relu6,
                                 name='conv2D_64_B')
            x = tf.layers.batch_normalization(x, training=False)
            x = tf.layers.max_pooling2d(x, [2, 2], strides=[2,2], padding='SAME')

        print("CONVB SHAPE: ", x.get_shape().as_list())
        
        x=tf.reshape(x,[batch_size,-1,1024])
        x=tf.transpose(x,[1,0,2])

        print("RESHAPE: ", x.get_shape().as_list())
        
        cells = []
        for _ in range(3):
            cell = tf.contrib.rnn.GRUCell(512)  # Or LSTMCell(num_units)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)
        
        output, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        
        print("GRU SHAPE: ", output.get_shape().as_list())
        
        with tf.variable_scope('Temporal_Average_Layer'):
            x = tf.reduce_mean(output,0)

        print("TEMPORAL AVG SHAPE: ", x.get_shape().as_list())

        with tf.variable_scope('Affine_Layer_A'):
            x = tf.layers.dense(x, 
                                units=256, 
                                activation = tf.nn.relu6,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))

        print("AFFINE SHAPE: ", x.get_shape().as_list())   
        print('-'*100)
        print()
    return x