import sys
import tensorflow as tf
import numpy as np
import librosa
from python_speech_features import fbank,delta
import scipy.io.wavfile as wave
from tensorflow.python.client import device_lib

def _parse_function(example_proto):
    ''' Function to parse tfrecords file '''
    feature = {'data': tf.VarLenFeature(tf.float32),
               'label':tf.FixedLenFeature([],tf.int64)}
    features = tf.parse_single_example(example_proto, features=feature)
    image = tf.sparse_tensor_to_dense(features['data'], default_value=0)
    label = tf.cast(features['label'], tf.int16)
    return image, label

def get_filterbanks(filename_placeholder, duration=8):
    ''' Returns filterbanks, delta1 and delta2 of input file '''
    def padding(audio,sr,duration=8):
        ''' Returns audio with padding '''
        nmax = sr*duration
        padlen = nmax - len(audio)
        audio = np.concatenate((audio, [0.0]*padlen))
        return audio

    def normalize_frames(m,epsilon=1e-12):
        ''' Normalizes features '''
        return np.array([(v - np.mean(v)) / max(np.std(v),epsilon) for v in m]).flatten()
    
    assert filename_placeholder.endswith('wav')
    window_fn= lambda x: np.hanning(x)
    sr,_ = wave.read(filename_placeholder)
    
    if not sr==16000:
        audio,_ = librosa.load(filename_placeholder, sr=sr, mono=True, duration=8)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    else:
        audio, sr = librosa.load(filename_placeholder, sr=sr, mono=True, duration=8)
    
    audio = audio.flatten()
    audio = padding(audio, sr,duration=8)
    
    filterbanks,_ = fbank(audio, samplerate=sr, winlen=0.025, 
                        winstep=0.01, nfilt=64, winfunc=window_fn)
    delta1 = delta(filterbanks, 1)
    delta2 = delta(delta1, 1)
    
    filterbanks = normalize_frames(filterbanks)
    delta1 = normalize_frames(delta1)
    delta2 = normalize_frames(delta2)
    
    features = np.concatenate((filterbanks, delta1, delta2))
    features = features.astype(np.float32)
    features = features.reshape((799,64,3))

    return features

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

