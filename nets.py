import tensorflow as tf

def model_softmax(x,batch_size,total_speakers):
    
    print('I/P shape ',x.get_shape())
    x=tf.reshape(x,[batch_size,-1,64,3])

    print('INPUT SHAPE ', x.get_shape())
    
    with tf.variable_scope('conv2D_A'):
        x=tf.layers.conv2d(x,filters=64,kernel_size=5, 
                           strides=(2,2),padding='SAME',kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),
                           activation=tf.nn.relu6,name='conv2D_64_A')
        print('Conv2D_A O/P shape ',x.get_shape())
        
    with tf.variable_scope('Batch_Norm_A'):
        x = tf.layers.batch_normalization(x)
        
    with tf.variable_scope('conv2D_B'):
        x=tf.layers.conv2d(x,filters=64,kernel_size=5, 
                           strides=(2,2),padding='SAME',kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(seed=0),
                           activation=tf.nn.relu6,name='conv2D_64')
        print('Conv2D_B O/P shape ',x.get_shape())
        
    with tf.variable_scope('Batch_Norm_B'):
        x = tf.layers.batch_normalization(x)
       
    x=tf.reshape(x,[batch_size,-1,1024])
    x=tf.transpose(x,[1,0,2])
    
    with tf.variable_scope('LSTM_A'):
        lstm_cell_A =tf.contrib.rnn.LayerNormBasicLSTMCell(512,dropout_keep_prob=0.9)
        x, _ = tf.nn.dynamic_rnn(lstm_cell_A, x, dtype=tf.float32)
        
    with tf.variable_scope('LSTM_B'):
        lstm_cell_B =tf.contrib.rnn.LayerNormBasicLSTMCell(512,dropout_keep_prob=0.9)
        x, _ = tf.nn.dynamic_rnn(lstm_cell_B, x, dtype=tf.float32)
        
    with tf.variable_scope('LSTM_C'):
        lstm_cell_C =tf.contrib.rnn.LayerNormBasicLSTMCell(512,dropout_keep_prob=0.9)
        x, states = tf.nn.dynamic_rnn(lstm_cell_C, x, dtype=tf.float32)
    
    print('LSTM O/P shape ',x.get_shape())

    with tf.variable_scope('Temporal_Average_Layer'):
        x=tf.reduce_mean(x,0)
        
    print('Temporal AVG O/P shape ',x.get_shape())
    
    with tf.variable_scope('Affine_Layer_A'):
        x=tf.layers.dense(x,units=256,activation=tf.nn.relu6, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
        
    print('Affine O/P shape ',x.get_shape())
    
    with tf.variable_scope('L2_Norm'):
        x = tf.nn.l2_normalize(x,1,name='embedding')
    
    print('Embedding Shape : ', x.get_shape())
    
    with tf.variable_scope('Softmax_Layer'):
        x=tf.layers.dense(x,units=total_speakers, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
        
    print('Softmax O/P shape ',x.get_shape())
    
    return x
