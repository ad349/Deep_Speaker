from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import time
import sys
import tensorflow as tf
import numpy as np
import importlib
import itertools
import argparse
# import facenet 
import bioid
#import lfw
import librosa
from python_speech_features import fbank,delta
import scipy.io.wavfile as wave
import models.bioid_cnn_lstm as network

#from nets import model_softmax as network
from utils import get_filterbanks #get_available_gpus

from tensorflow.python.ops import data_flow_ops

from six.moves import xrange # Python 2.7

def main(args):
    # This loads the graph for the model
    # In our case we load if from a py file
    
    # network = importlib.import_module(args.model_def)
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    # Write arguments to a text file
    bioid.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
    
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    bioid.store_revision_info(src_path, log_dir, ' '.join(sys.argv))
    
    np.random.seed(seed=args.seed)
    train_set = bioid.get_dataset(args.data_dir)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))
    
    # This part reads test data for testing
    #if args.lfw_dir:
    #    print('LFW directory: %s' % args.lfw_dir)
    #    # Read the file containing the pairs used for testing
    #    pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
    #    # Get the paths for the corresponding images
    #    lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
    
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
  
        #available_gpus=get_available_gpus()
        #num_clones=len(available_gpus)
        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        wave_paths_placeholder = tf.placeholder(tf.string, shape=(None,3), name='wave_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None,3), name='labels')
        # dp = tf.constant(args.duration, name='duration')
        # filename_placeholder = tf.placeholder(tf.string)
        
        # Should also try PaddingFIFO Queue
        input_queue = data_flow_ops.FIFOQueue(capacity=110000,
                                    dtypes=[tf.string, tf.int64],
                                    shapes=[(3,), (3,)],
                                    shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([wave_paths_placeholder, labels_placeholder])
        
        nrof_preprocess_threads = args.nrof_preprocess_threads
        waves_and_labels = []
        
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            waves = []
            for filename in tf.unstack(filenames):
                wave = tf.py_func(get_filterbanks, [filename], tf.float32)
                wave.set_shape((args.nframes, args.nfilt, 3))
                waves.append(wave)
            waves_and_labels.append([waves, label])
            
        wave_batch, labels_batch = tf.train.batch_join(
            waves_and_labels,
            batch_size=batch_size_placeholder,
            shapes=[(args.nframes, args.nfilt, 3), ()],
            enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        wave_batch = tf.identity(wave_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'label_batch')

        # Build the inference graph
        logits = network.inference(wave_batch, batch_size_placeholder, args.keep_probability, 
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size, weight_decay=args.weight_decay)
        
        embeddings = tf.nn.l2_normalize(logits, 1, 1e-10, name='embedding')

        # L2 Normalize embeddings in the model definition.
        # embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # Split embeddings into anchor, positive and negative and calculate triplet loss
        # Need to confirm the shape here
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3,args.embedding_size]), 3, 1)
        triplet_loss = bioid.triplet_loss(anchor, positive, negative, args.alpha)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = bioid.train(total_loss, global_step, args.optimizer,
            learning_rate, args.moving_average_decay, tf.global_variables())

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction, allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True))        
        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:True})

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                saver.restore(sess, os.path.expanduser(args.pretrained_model))

            # Training and validation loop
            epoch = 0
            try:
                print('Max number of Epochs: ',args.max_nrof_epochs)
                while epoch < args.max_nrof_epochs:
                    step = sess.run(global_step, feed_dict=None)
                    #epoch = step // args.epoch_size
                    epoch += 1
                    # Train for one epoch
                    train(args, sess, train_set, epoch, wave_paths_placeholder,
                          labels_placeholder, labels_batch, batch_size_placeholder,
                          learning_rate_placeholder, phase_train_placeholder, enqueue_op,
                          input_queue, global_step, embeddings, total_loss, train_op,
                          summary_op, summary_writer, args.learning_rate_schedule_file,
                          args.embedding_size, anchor, positive, negative, triplet_loss)
                    
                    # Save variables and the metagraph if it doesn't exist already
                    save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                    # Evaluate on LFW
                    # Add Evaluate function from facenet
                    # if args.lfw_dir:
                    #     evaluate(sess, lfw_paths, embeddings, labels_batch, wave_paths_placeholder, labels_placeholder, 
                    #             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, 
                    #                enqueue_op, actual_issame, args.batch_size, 
                    #             args.lfw_nrof_folds, log_dir, step, summary_writer, args.embedding_size)
                print('All Epochs Finished!')
                print('Stopping all threads')
                sess.run(input_queue.close())
                coord.request_stop()
                coord.join(threads)
                sess.close()
            except tf.errors.OutOfRangeError:
                print("Done Training -- epoch limit reached or the coordinator signalled to stop.")
                coord.request_stop()
                coord.join(threads)
                sess.close()
    return model_dir


def train(args, sess, dataset, epoch, wave_paths_placeholder, labels_placeholder,
          labels_batch, batch_size_placeholder, learning_rate_placeholder,
          phase_train_placeholder, enqueue_op, input_queue, global_step, embeddings,
          loss, train_op, summary_op, summary_writer, learning_rate_schedule_file, 
          embedding_size, anchor, positive, negative, triplet_loss):
    
    batch_number = 0
    epoch_loss = 0
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = bioid.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    while batch_number < args.epoch_size:
        # Sample people randomly from the dataset
        wave_paths, num_per_class = sample_people(dataset, args.people_per_batch, args.waves_per_person)

        start_time = time.time()
        nrof_examples = args.people_per_batch * args.waves_per_person
        labels_array = np.reshape(np.arange(nrof_examples),(-1,3))
        wave_paths_array = np.reshape(np.expand_dims(np.array(wave_paths),1), (-1,3))
        sess.run(enqueue_op, {wave_paths_placeholder: wave_paths_array, labels_placeholder: labels_array})
        emb_array = np.zeros((nrof_examples, embedding_size))
        nrof_batches = int(np.ceil(nrof_examples / args.batch_size))
        print('Number of batches for forward pass: ', nrof_batches)
        print('Running forward pass on sampled waves took: ', end='')
        # Run this on GPUs, no need to average the losses as no loss is calculated here
        for i in range(nrof_batches):
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size, 
                learning_rate_placeholder: lr, phase_train_placeholder: True})
            emb_array[lab,:] = emb
        print('%.3f' % (time.time()-start_time))

        # Select triplets based on the embeddings
        print('Selecting suitable triplets for training')
        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class,
                                                                    wave_paths, args.people_per_batch, args.alpha)
        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' % 
            (nrof_random_negs, nrof_triplets, selection_time))

        # Perform training on the selected triplets
        nrof_batches = int(np.ceil(nrof_triplets*3/args.batch_size))
        triplet_paths = list(itertools.chain(*triplets))
        labels_array = np.reshape(np.arange(len(triplet_paths)),(-1,3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths),1), (-1,3))
        sess.run(enqueue_op, {wave_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})
        nrof_examples = len(triplet_paths)
        train_time = 0
        i = 0
        emb_array = np.zeros((nrof_examples, embedding_size))
        loss_array = np.zeros((nrof_triplets,))
        epoch_loss = []
        #summary = tf.Summary()
        # Run this on multi GPUs and update average loss
        print('Number of batches to update loss :', nrof_batches)
        while i < nrof_batches:
            start_time = time.time()
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr, phase_train_placeholder: True}
            err, _, step, emb, lab = sess.run([loss, train_op, global_step, embeddings, labels_batch], feed_dict=feed_dict)
            epoch_loss.append(err)
            emb_array[lab,:] = emb
            loss_array[i] = err
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.6f' % 
                (epoch, batch_number+1, args.epoch_size, duration, err)) # changed args.epoch_size to nrof_batches
            batch_number += 1
            i += 1
            train_time += duration
            #summary.value.add(tag='loss/step', simple_value=err)
        # Add validation loss and accuracy to summary
        #pylint: disable=maybe-no-member
        #summary.value.add(tag='time/selection', simple_value=selection_time)
    avg_epoch_loss = sum(epoch_loss)/len(epoch_loss)
    print('-'*100)
    print('Epoch: [%d]\tAvg Loss %2.6f' % (epoch, avg_epoch_loss))
    print('-'*100)
    #summary.value.add(tag='loss/epoch_loss', simple_value=avg_epoch_loss)
    #summary_writer.add_summary(summary, step)
    return True


def select_triplets(embeddings, nrof_waves_per_class, wave_paths, people_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    # Deep Face Recognition
    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.

    for i in xrange(people_per_batch):
        nrof_waves = int(nrof_waves_per_class[i])
        for j in xrange(1,nrof_waves):
            a_idx = emb_start_idx + j - 1
            # It is L2 distance as of now, but could change this to cosine similarity
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_waves): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_waves] = np.NaN # np.Nan
                #all_neg = np.where(neg_dists_sqr<pos_dist_sqr)[0]
                all_neg = np.where(np.logical_or(
                    neg_dists_sqr<pos_dist_sqr, np.logical_and(
                        pos_dist_sqr<neg_dists_sqr,neg_dists_sqr<pos_dist_sqr+alpha)))[0]
                # all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]
                # Facenet/Same as Deep Speaker
                #all_neg = np.where(neg_dists_sqr-pos_dist_sqr < alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((wave_paths[a_idx], wave_paths[p_idx], wave_paths[n_idx]))
                    #print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' % \
                    #      (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, 
                    #       neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, 
                    #       emb_start_idx))
                    trip_idx += 1
                num_trips += 1
        emb_start_idx += nrof_waves

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)


def sample_people(dataset, people_per_batch, waves_per_person):
    nrof_waves = people_per_batch * waves_per_person
  
    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    
    i = 0
    wave_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(wave_paths)<nrof_waves:
        class_index = class_indices[i]
        nrof_waves_in_class = len(dataset[class_index])
        wave_indices = np.arange(nrof_waves_in_class)
        np.random.shuffle(wave_indices)
        nrof_waves_from_class = min(nrof_waves_in_class, waves_per_person, nrof_waves-len(wave_paths))
        idx = wave_indices[0:nrof_waves_from_class]
        wave_paths_for_class = [dataset[class_index].wave_paths[j] for j in idx]
        sampled_class_indices += [class_index]*nrof_waves_from_class
        wave_paths += wave_paths_for_class
        num_per_class.append(nrof_waves_from_class)
        i+=1
  
    return wave_paths, num_per_class

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate

def parse_arguments(argv):
    # Comments
    # Module bioid - A Python script
    # Using NCHW coz its faster on GPUs and more intuitive
    # Better to make an object class for model exactly how its done in facenet
    # add write_arguments_to_file
    # args.model_def - Python script where our model is defined
    # args.logs_dir
    # args.models_base_dir
    # args.seed
    # args.data_dir
    # args.pretrained_model - We keep this as previous checkpoint to load
    # args.lfw_dir - file containing test data pairs
    # args.tfrecords
    # args.buffer - Shuffle buffer size
    # args.batch_size
    # args.nfilt - number of filterbanks
    # args.nframes - int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1
    # args.duration - Max duration of audio to be used
    # args.keep_probability - Dropout keep probability
    # args.embedding_size
    # args.alpha - Embedding similarity margin
    # args.learning_rate_decay_epochs
    # args.epoch_size
    # args.learning_rate_decay_factor
    # args.optimizer
    # args.moving_average_decay
    # args.gpu_memory_fraction
    # args.max_nrof_epochs
    # args.learning_rate_schedule_file
    # args.people_per_batch
    # args.waves_per_person
    # args.learning_rate
    # TFRECORDS 
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_def', type=str, 
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.bioid_cnn_lstm')
    parser.add_argument('--logs_dir', type=str, 
        help='Directory where to write event logs.', default='./logs/bioid')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='./models/bioid')
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--data_dir', type=str, 
        help='Path to the data directory containing aligned face patches.',
        default='~/datasets/external')
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model checkpoint before training starts.')
    parser.add_argument('--batch_size', type=int,
        help='Number of waves to process in a batch.', default=90)
    parser.add_argument('--nfilt', type=int,
        help='Number of filter banks', default=64)
    parser.add_argument('--nframes', type=int,
        help='Number of frames in each feature int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1', default=799)
    parser.add_argument('--duration', type=int,
        help='Duration of audio in seconds to use', default=8)
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=0.1)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=256)
    parser.add_argument('--alpha', type=float,
        help='Positive to negative triplet distance margin.', default=0.15)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=10)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=0.96)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.7)
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=100)
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')
    parser.add_argument('--people_per_batch', type=int,
        help='Number of people per batch.', default=45)
    parser.add_argument('--waves_per_person', type=int,
        help='Number of wave files per person.', default=50)
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--weight_decay', type=float, 
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--nrof_preprocess_threads', type=int, help='Preprocessing threads', default=10)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
