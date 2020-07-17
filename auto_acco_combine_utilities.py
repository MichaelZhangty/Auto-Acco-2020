import math
import numpy as np
import pretty_midi
# import analyse
# from madmom.features.onsets import CNNOnsetProcessor
from scipy.integrate import quad
import aubio
import matplotlib.pyplot as plt
import wave
import statsmodels.api as sm

import mido
from mido import MidiFile
import pretty_midi
import threading
import pretty_midi
# import scikits.audiolab
import pyaudio
# import analyse
import time
import copy
from scipy.integrate import quad
import wave
import numpy as np
import math
import sys
import pretty_midi
import matplotlib.pyplot as plt
from madmom.features.onsets import CNNOnsetProcessor
import os
import librosa
import statsmodels.api as sm
import aubio
# from auto_acco_combine_utilities import *
# write a clean version of score_following
# one set up function
# one while running function 
# shared is sQueue
# get time axis from midi file
def get_time_axis(resolution, filename):
    resolution = 0.01
    midi_data = pretty_midi.PrettyMIDI(filename)
    axis_start_time = 0
    axis_end_time = midi_data.instruments[0].notes[-1].end
    # print("end_time" + str(axis_end_time))
    scoreLen = int(math.ceil(axis_end_time/resolution)) + 1
    size = int((axis_end_time - axis_start_time) / resolution + 1)
    axis = np.linspace(axis_start_time, axis_end_time, size)
    score_midi = np.full(scoreLen,-1)# no sound = -1
    raw_score_midi = np.full(scoreLen,-1)
    score_onsets = np.zeros(scoreLen)
    onsets = []
    for note in midi_data.instruments[0].notes:
        start = int(math.floor(note.start / resolution))
        end = int(math.ceil(note.end / resolution)) + 1
        if start < len(score_onsets):
            score_onsets[start] = 1
        onsets.append(start)
        for j in range(start, end):
            if j < len(score_midi):
                score_midi[j] = note.pitch%12 # regulate to 12 pitch
                raw_score_midi[j] = note.pitch
    # plot to check
    # plt.plot(score_midi[0:500])
    # plt.show()
    return score_midi, axis, score_onsets, onsets, raw_score_midi

def score_midi_modification(score_midi):
    # change score_midi to make every pitch last for at most 1.5s
    temp_pitch = -1
    count = 1
    for i in range(len(score_midi)):
        if score_midi[i] == temp_pitch:
            count += 1
            if count >= 150:
                score_midi[i] = -1
        else:
            temp_pitch = score_midi[i]
            count = 1
    # plt.plot(score_midi)
    # plt.show()
    return score_midi

def pitch_detection_aubio(data,size):
    CHUNK = 1024
    # CHUNK = 1412
    pitch_detector = aubio.pitch('yinfft', CHUNK*size, CHUNK*size, 44100)
    pitch_detector.set_unit('midi')
    pitch_detector.set_tolerance(0.75)
    samps = np.fromstring(data, dtype=np.int16)
    samps = np.true_divide(samps, 32768, dtype=np.float32)
    pitch = pitch_detector(samps)[0]
    if pitch > 84 or pitch < 40:
        return -1
    else:
        return pitch

def tempo_estimate(elapsed_time, cur_pos, old_pos,Rc,resolution):
    return float(cur_pos - old_pos) * Rc * resolution / elapsed_time

def compute_f_I_given_D(fsource, f_I_J_given_D, cur_pos, scoreLen,window_size=200):
    left = max(0, cur_pos - window_size)
    right = min(scoreLen, cur_pos + window_size)
    f_I_given_D = np.zeros(scoreLen)
    fsource_w = fsource[left:right]
    f_I_J_given_D_w = f_I_J_given_D[:right - left]
    f_I_given_D_w = np.convolve(fsource_w, f_I_J_given_D_w)
    f_I_given_D_w = f_I_given_D_w / sum(f_I_given_D_w)
    if left + len(f_I_given_D_w) > scoreLen:
        end = scoreLen
    else:
        end = left + len(f_I_given_D_w)
    f_I_given_D[left:end] = f_I_given_D_w[:(end - left)]
    # plt.plot(f_I_given_D[:50])
    # plt.show()
    # plt.plot(fsource[:50])
    # plt.show()
    return f_I_given_D

def compute_f_I_J_given_D(score_axis, estimated_tempo, elapsed_time, beta,alpha,Rc,no_move_flag):
    # if no_move_flag:
    #     print("no move")
    if estimated_tempo > 0:
        rateRatio = float(Rc) / float(estimated_tempo)
    else:
        rateRatio = Rc / 0.00001
    rateRatio = 1/rateRatio
    sigmaSquare = math.log(float(1) / float(alpha * elapsed_time) + 1)
    sigma = math.sqrt(sigmaSquare)
    tmp1 = 1 / (score_axis * sigma * math.sqrt(2 * math.pi))
    tmp2 = (np.log(score_axis) - math.log(rateRatio * elapsed_time) + beta * sigmaSquare)
    tmp2 = np.exp(-tmp2 * tmp2 / (2 * sigmaSquare))
    distribution = tmp1 * tmp2
    distribution[score_axis <= 0] = 0

    distribution = distribution / sum(distribution)
    # for debug
    # plt.plot(distribution[:15])
    # plt.show()
    return distribution



def compute_f_V_given_I(pitch, pitches, scoreLen, score_midi, onset_prob, score_onsets, alpha, w1, w2, w3,cur_pos,
std=1,WINSIZE = 1,WEIGHT=[0.5]):
    # weight = 0.5 original
    reverse_judge = False
    f_V_given_I = np.zeros(scoreLen)
    sims = np.zeros(scoreLen)
    if pitch == -1:
        pitch = -1
    elif len(pitches) > WINSIZE:
        if pitch > 11.5:
            pitch_reverse = pitch - 12
            pitch_reverse = pitch_reverse - np.dot(pitches[-1 - WINSIZE:-1], WEIGHT)
            reverse_judge = True
        elif pitch < 0.5:
            pitch_reverse = pitch + 12
            pitch_reverse = pitch_reverse - np.dot(pitches[-1 - WINSIZE:-1], WEIGHT)
            reverse_judge = True
        pitch = pitch - np.dot(pitches[-1 - WINSIZE:-1], WEIGHT)

    # to check for two tempo at most per pitch
    # each i represent 0.01s
    window_size = 200
    left = max(0, cur_pos - window_size)
    right = min(scoreLen, cur_pos + window_size)
   
    for i in range(left,right):
        if pitch == -1 or score_midi[i] == -1:
            score_pitch = score_midi[i]
        elif i >= WINSIZE:
            score_pitch = score_midi[i] - np.dot(score_midi[i - WINSIZE:i], WEIGHT)
        else:
            score_pitch = score_midi[i]
        score_onset = score_onsets[i]
        if pitch == -1:
            if score_pitch == -1:
                f_V_given_I[i] = 0.1
            else:
                f_V_given_I[i] = 0.00000000001
        elif score_pitch == -1:
                f_V_given_I[i] = 0.00000000001
        else:
            if reverse_judge and abs(pitch-score_pitch) > abs(pitch_reverse-score_pitch):
                pitch = pitch_reverse
            f_V_given_I[i] = math.pow(
                math.pow(normpdf(pitch, score_pitch, std), w1) * math.pow(similarity(onset_prob, score_onset), w2), w3)

    return f_V_given_I

def create_gate_mask(cur_pos, scoreLen):
    mask = np.zeros(scoreLen)

    for i in range(-50, 51):
        if cur_pos + i < scoreLen and cur_pos + i >= 0:
            mask[cur_pos + i] = 1
    # plt.plot(range(scoreLen),mask, color='black')
    # plt.show()source
    return mask

def normpdf(x, mean, sd=1):
    var = float(sd) ** 2
    pi = 3.1415926
    denom = (2 * pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom

def diff(score):
    diff_score = np.zeros(len(score))
    for i in range(len(score) - 1):
        diff_score[i] = score[i + 1] - score[i]
    return diff_score

def sigmoid(x):
    return 1 / (1 + math.e ** -x)

def similarity(onset_prob, score_onset):
    sim = float(min(onset_prob, score_onset) + 1e-6) / (max(onset_prob, score_onset) + 1e-6)
    return sim



def compute_tempo_ratio_weighted(b0, t0, s0, l,timeQueue,beat_back,confidence_queue):
    t1 = timeQueue[-1]
    b = b0 + (t1 + l - t0) * s0
    tn = t1 + l
    # print "latency l %f---------------------------------"%l
    bn = b
    te = timeQueue[-2]
    be = len(timeQueue) - 5
    x = timeQueue[-beat_back:]
    y = list(range(len(timeQueue) - beat_back - 3, len(timeQueue) - 3))
    confidence_block = confidence_queue[(len(timeQueue) - beat_back):len(timeQueue)]
    x = sm.add_constant(x)
    if y[0] == 0:
        wls_model = sm.WLS(y, x)
        results = wls_model.fit()
        se = results.params[1]
    else:
        wls_model = sm.WLS(y, x, weights=confidence_block)
        results = wls_model.fit()
        se = results.params[1]
    # print "------------------------------sk-----------------------"
    # print se
    sn = (float(4) / (te * se - tn * se - be + bn + 4)) * se
    return bn, tn, sn
