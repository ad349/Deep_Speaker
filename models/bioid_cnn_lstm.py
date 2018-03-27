from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Input Shape: [-1, 799, 64, 3]
# Layer1 Shape: [-1, 400, 32, 32]
# Layer2 Shape: [-1, 200, 16, 64]
# Layer3 Shape: [-1, 100, 8, 128]
# Layer3 Shape: [-1, 50, 4, 256]

def clipped_relu(x):
    return tf.minimum(tf.maximum(x, 0), 20, name='clipped_relu')

def identity_block(inp_tensor, kernel_size, filters, weight_decay, phase_train, seed, block, stage):
    block_name = 'res{}_{}_{}'.format(filters, stage, block)
    
    x = tf.layers.conv2d(inp_tensor,
                         filters=filters,
                         kernel_size=kernel_size,
                         strides=1,
                         padding='SAME',
                         data_format='channels_last',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed), 
                         activation=None, 
                         use_bias=True, 
                         bias_initializer=tf.zeros_initializer(), 
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                         bias_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                         name=block_name+'_conv2a')
    
    x = tf.layers.batch_normalization(x, training=phase_train, name=block_name+'_batch_norm_a')
    
    x = clipped_relu(x)
    
    x = tf.layers.conv2d(inp_tensor,
                         filters=filters,
                         kernel_size=kernel_size,
                         strides=1,
                         padding='SAME',
                         data_format='channels_last',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed), 
                         activation=None, 
                         use_bias=True, 
                         bias_initializer=tf.zeros_initializer(), 
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                         bias_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                         name=block_name+'_conv2b')
    
    x = tf.layers.batch_normalization(x, training=phase_train, name=block_name+'_batch_norm_b')
    
    x = tf.add_n([x, inp_tensor], name='_add')
    
    x = clipped_relu(x)
    
    return x

def conv_identity_block(inp_tensor, filters, weight_decay, phase_train, seed, stage):
    conv_name = 'conv-res{}_{}'.format(filters, stage)
    
    x = tf.layers.conv2d(inp_tensor,
                         filters=filters,
                         kernel_size=5,
                         strides=2,
                         padding='SAME',
                         data_format='channels_last',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed), 
                         activation=None, 
                         use_bias=True, 
                         bias_initializer=tf.zeros_initializer(), 
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                         bias_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                         name=conv_name+'_conv2a')
    
    x = tf.layers.batch_normalization(x, training=phase_train, name=conv_name+'_batch_norm_a')
    
    x = clipped_relu(x)
    
    for i in range(3):
        o = identity_block(x, kernel_size=3, 
                           filters=filters, weight_decay=weight_decay, 
                           phase_train=phase_train, seed=seed, block=i, stage=stage)
    return o

    
def inference(inp_tensor, batch_size, keep_probability, phase_train, weight_decay, embedding_size):
    rnd = np.random.randint(100, 1000, 1)
    print()
    print('-'*100)
    print("INPUT SHAPE: ", inp_tensor.get_shape().as_list())
    with tf.name_scope('Model'):
        with tf.variable_scope('Block_1'):
            x = conv_identity_block(inp_tensor, 64, weight_decay, phase_train, seed=rnd, stage=1)
        print('BLOCK 1 SHAPE: ', x.get_shape().as_list())
        
        with tf.variable_scope('Block_2'):
            x = conv_identity_block(x, 128, weight_decay, phase_train, seed=rnd, stage=2)
        print('BLOCK 2 SHAPE: ', x.get_shape().as_list())
            
        with tf.variable_scope('Block_3'):
            x = conv_identity_block(x, 256, weight_decay, phase_train, seed=rnd, stage=3)
        print('BLOCK 3 SHAPE: ', x.get_shape().as_list())
            
        with tf.variable_scope('Block_4'):
            x = conv_identity_block(x, 512, weight_decay, phase_train, seed=rnd, stage=4)
        print('BLOCK 4 SHAPE: ', x.get_shape().as_list())
            
        with tf.variable_scope('Reshape'):
            x = tf.reshape(x, [batch_size, -1, 2048])
        print('RESHAPED: ', x.get_shape().as_list())
            
        with tf.variable_scope('Temporal_Mean'):
            x = tf.reduce_mean(x, axis=1)
        print('TEMPORAL MEAN: ', x.get_shape().as_list())
            
        with tf.variable_scope('Affine'):
            x = tf.layers.dense(x, units=embedding_size)
        print('AFFINE: ', x.get_shape().as_list())
            
        with tf.variable_scope('L2_Normalize'):
            x = tf.nn.l2_normalize(x, 1, 1e-10, name='embedding')
        print('L2 NORMALIZED: ', x.get_shape().as_list())
        
        print('-'*100)
        print()       
    return x