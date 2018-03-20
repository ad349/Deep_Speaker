"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import bioid
#import lfw
import glob
import os
import sys
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from models.bioid_cnn_lstm import inference
from utils import get_filterbanks

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            # Read the file containing the pairs used for testing
            #pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))

            # Get the paths for the corresponding images
            #paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

            # Load the model
            bioid.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("L2_Norm/embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            #image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            #image_size = args.image_size
            #embedding_size = embeddings.get_shape()[1]
            
            path = args.test_dir + '**.wav'
            folders = glob.glob(path)
            df = pd.DataFrame(folders, columns=['filename'])
            df['speakerid'] = df.filename.apply(lambda x: x.split('/')[-1].split('@')[0])
            df['count'] = df.groupby(df['speakerid'])['speakerid'].transform('count')
            df = df[df['count'] >= 5]
            speaker_ids = df['speakerid'].unique()
            imposter_id = speaker_ids[:len(speaker_ids)/2]
            user_id = speaker_ids[len(speaker_ids)/2:]
            
            batch_size = args.batch_size
            num_enrollments= args.nrof_enrollments
            
            enrollment_df = pd.DataFrame()
            user_df = pd.DataFrame()
            
            for i in user_id:
                user_temp = df[df['speakerid']==i].reset_index(drop=True)
                enrollment_df = pd.concat([enrollment_df,user_temp[:num_enrollments]],axis=0)
                user_df = pd.concat([user_df,user_temp[num_enrollments:]],axis=0)
            
            enrollment_df = enrollment_df.reset_index(drop=True)
            user_df = user_df.reset_index(drop=True)
            imposter_df = pd.DataFrame()
            
            for i in imposter_id:
                imposter_temp = df[df['speakerid']==i].reset_index(drop=True)
                imposter_df = pd.concat([imposter_df,imposter_temp[:num_enrollments]],axis=0)

            imposter_df = imposter_df.reset_index(drop=True)
            eval_xs_enrollment = enrollment_df['filename'].apply(lambda x: get_filterbanks(x))
            eval_xs_user = user_df['filename'].apply(lambda x:get_filterbanks(x))
            eval_xs_imposter = imposter_df['filename'].apply(lambda x:get_filterbanks(x))
            
            embd_enrollment=[]
            embd_user=[]
            embd_imposter=[]
            
            print('Genrating Embeddings !!')
            for eval_x in eval_xs_enrollment:
                embd_enrollment.append(sess.run(embeddings,feed_dict={waves_placeholder:waves, phase_train_placeholder:False}))
            for eval_x in eval_xs_user:
                embd_user.append(sess.run(embeddings,feed_dict={waves_placeholder:waves, phase_train_placeholder:False}))
            for eval_x in eval_xs_imposter:
                embd_imposter.append(sess.run(embeddings,feed_dict={waves_placeholder:waves, phase_train_placeholder:False}))
                
            enrollment_df['embedding']=embd_enrollment
            user_df['embedding']=embd_user
            imposter_df['embedding']=embd_imposter
        
            tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array, 
                actual_issame, nrof_folds=args.lfw_nrof_folds)

            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

            auc = metrics.auc(fpr, tpr)
            print('Area Under Curve (AUC): %1.3f' % auc)
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            print('Equal Error Rate (EER): %1.3f' % eer)
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--batch_size', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=1)
    parser.add_argument('--nrof_enrollments
        help='Number of folds to use for cross validation. Mainly used for testing.', default=1)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))