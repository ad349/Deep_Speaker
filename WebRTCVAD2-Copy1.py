import collections
import contextlib
import sys
import os
import wave
import argparse
import webrtcvad
from glob import glob
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import cpu_count


def ready_df(input,cpu):
    return np.array_split(pd.read_csv(input), cpu)


def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    min_padding_frames = int(4000/frame_duration_ms)
    max_padding_frames = int(7000/frame_duration_ms)
    ring_buffer = collections.deque(maxlen=max_padding_frames)
    triggered = False
    voiced_frames = []
    for frame in frames:
        if not triggered:
            ring_buffer.append(frame)
            num_voiced = len([f for f in ring_buffer
                              if vad.is_speech(f.bytes, sample_rate)])
            if num_voiced > min_padding_frames:
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append(frame)
            num_unvoiced = len([f for f in ring_buffer
                                if not vad.is_speech(f.bytes, sample_rate)])
            if (num_unvoiced > 0.001 * ring_buffer.maxlen) or (len(voiced_frames) >= max_padding_frames):
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


def brain(df_list,agg,frame,padding,output):
    Process = os.getpid()
    try:
        for i in range(len(df_list)):                
            abs_wav_files = df_list.values[i][0]
            abs_filename = os.path.splitext(os.path.basename(abs_wav_files))[0]
            audio, sample_rate = read_wave(abs_wav_files)
            vad = webrtcvad.Vad(int(agg))
            frames = frame_generator(frame, audio, sample_rate)
            frames = list(frames)
            segments = vad_collector(sample_rate, frame, padding, vad, frames)
            for j, segment in enumerate(segments):
                path = os.path.join(output,abs_filename) + '-%0002d.wav' % (j+1,)
                print("Process {}: Created file {}".format(Process,path))
                write_wave(path,segment,sample_rate)
    except EOFError as e:
        print("Empty file {}".format(abs_wav_files_filename))
        pass

    
def main(args):
    if len(sys.argv) < 10:
        sys.stderr.write(
            'Usage: example.py [--frame | --padding | --input | --output | --aggressiveness] <path to wav file>\n')
        sys.exit(1)
    input_p = os.path.abspath(args.input)
    output_p = os.path.abspath(args.output)
    cpu = 2*cpu_count()**2
    frame = args.frame
    padding = args.padding
    agg = args.agg
    
    if not os.path.exists(input_p):
        sys.stderr.write('Input folder does not exist!')
        sys.exit(1)
    if not os.path.exists(output_p):
        os.makedirs(output_p)
    df_list = ready_df(input_p,cpu)
    processes = [mp.Process(target=brain, args=(x, agg,frame,padding,output_p)) for x in df_list]
    print("Number of processes : %s " % len(processes))
    [p.start() for p in processes]
    [p.join() for p in processes]

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame',type=int,default=10,help='Duration of frame')
    parser.add_argument('--padding',type=int,default=5000,help='Padding Duration')
    parser.add_argument('--input', type=str, default='',help='Directory of Wav files')
    parser.add_argument('--output',type=str,default='',help='Save path')
    parser.add_argument('--agg', type=int, default='',help='aggressiveness between 1 to 3')
    args = parser.parse_args()
    main(args)
