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
    midi_data = pretty_midi.PrettyMIDI(filename)
    axis_start_time = 0
    axis_end_time = midi_data.instruments[0].notes[-1].end
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
                score_midi[j] = note.pitch - note.pitch / 12 * 12 # regulate to 12 pitch
                raw_score_midi[j] = note.pitch
    # plot to check
    # plt.plot(score_midi)
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

def compute_f_I_given_D(fsource, f_I_J_given_D, cur_pos, scoreLen,window_size=1000):
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
    window_size = 500
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

# audio_name = "audio4"
# AUDIOFILE = 'audio/{}.wav'.format(audio_name)
# midi_name = "midi4"
# MIDIFILE = 'midi/{}.mid'.format(midi_name)
# resolution = 0.01
# midi_file = MIDIFILE
# score_midi, score_axis, score_onsets, onsets, raw_score_midi = get_time_axis(resolution,midi_file)
# score_midi = score_midi_modification(score_midi)
# scoreLen = len(score_axis)
# fsource = np.zeros(scoreLen)
# confidence = np.zeros(scoreLen)

def score_following(audio_file,audio_end_time,Rc,resolution,sQueue,fsource,scoreLen,confidence,score_midi,latency_end):

    # set up for audio input 
    CHUNK = 1024
    time_int = float(CHUNK) / 44100
    performance_start_time = 0
    cur_time = performance_start_time
    old_time = performance_start_time
    tempo_estimate_elapsed_time = 0
    tempo_estimate_elapsed_time2 = 0
    wf = wave.open(audio_file, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    n_frames = int(performance_start_time * wf.getframerate())
    wf.setpos(n_frames)
    start_time = time.clock()
    count_cut = 1 
    datas = []
    estimated_tempo = Rc
    confidence_record_check = 0
    time_list_for_beat = [1,2,3,4,5]
    confidence_queue = [0.001, 0.001, 0.001, 0.001, 0.001]
    temp_downbound = 0.7
    temp_upbound = 1.3
    no_move_flag = False
    fsource[0] = 1
    cur_pos = 0
    old_pos = 0
    alpha = 10
    pitches = []
    onset_idx = 0
    b0 = 1
    t0 = time_list_for_beat[-1]
    s0 = float(1) / (time_list_for_beat[-1] - time_list_for_beat[-2])
    sQueue.append(s0)
    beat_back = 4
    while wf.tell() < wf.getnframes():
        while time.clock() - start_time < count_cut * time_int:
            pass
        count_cut += 1
        data = wf.readframes(CHUNK)
        datas.append(data)
        if len(datas) >= 3:
            c_data = datas[-3:]
            c_data = ''.join(c_data)
            pitch = pitch_detection_aubio(c_data,3)
        else:
            c_data = data
            pitch = pitch_detection_aubio(c_data,1)

        onset_detector = aubio.onset("default", CHUNK, CHUNK, 44100)
        data_onset = np.fromstring(data, dtype=np.int16)
        data_onset = np.true_divide(data_onset, 32768, dtype=np.float32)
        if len(data_onset) == 1024 and onset_detector(data_onset):
            onset_prob = onset_detector.get_last() / 44100

        tempo_estimate_elapsed_time += time_int
        tempo_estimate_elapsed_time2 += time_int
        cur_time += time_int

        if cur_time > audio_end_time:
            break

        if pitch != -1:
            pitch = pitch - int(pitch) / 12 * 12
        pitches.append(pitch)
        # print("pitch detected is " + str(pitch))
        elapsed_time = cur_time - old_time
        tempo = estimated_tempo

        # record the beat time
        if tempo * tempo_estimate_elapsed_time2 > 1 * Rc:
            confidence_record_check = 1
            time_list_for_beat.append(time_list_for_beat[-1] + tempo_estimate_elapsed_time2)  # list can be improved
            tempo_estimate_elapsed_time2 = 0
        
        if tempo * tempo_estimate_elapsed_time > 2 * Rc:
            tempo = tempo_estimate(tempo_estimate_elapsed_time, cur_pos, old_pos,Rc,resolution)
            # with bound for tempo
            if tempo / float(Rc) < temp_downbound:
                tempo = Rc * temp_downbound
            elif tempo / float(Rc) > temp_upbound:
                tempo = Rc * temp_upbound
            tempo_estimate_elapsed_time = 0 
            estimated_tempo = tempo
            old_pos = cur_pos

            # change the tempo for auto acco
            if latency_end == -1:
                l = 0.1
            else:
                l = max(0, latency_end - time.clock())
            b0, t0, s0 = compute_tempo_ratio_weighted(b0, t0, s0, l,time_list_for_beat,beat_back,confidence_queue)
            sQueue.append(s0)
            print(sQueue)


        if fsource[cur_pos] > 0.1:
            beta = 0.5
            w1 = 0.95
            w2 = 0.05
            w3 = 0.5
            # bigger w3 means pitch weight more
        else:
            beta = 0.5
            w1 = 0.7
            w2 = 0.3
            w3 = 0.3
        
        f_I_J_given_D = compute_f_I_J_given_D(score_axis, tempo, elapsed_time, beta,alpha,Rc,no_move_flag)
        f_I_given_D = compute_f_I_given_D(fsource, f_I_J_given_D, cur_pos, scoreLen)
        if no_move_flag:
            f_I_given_D = fsource
        cur_pos = np.argmax(f_I_given_D)
        # F_V_give_I is the p(o|st)
        # F_I_give_D is the p(st)
        # f_source = p(st+1)
        f_V_given_I = compute_f_V_given_I(pitch, pitches, scoreLen, score_midi, onset_prob, score_onsets, alpha,
                                                 w1, w2, w3,cur_pos)
        fsource = f_V_given_I * f_I_given_D
        gate_mask = create_gate_mask(cur_pos, scoreLen)
        fsource = fsource * gate_mask
        fsource = fsource / sum(fsource)
        cur_pos = np.argmax(fsource)
        # print("cur_pos is " + str(cur_pos))

        # for suddenly slient while singing in the middle       
        if pitch == -1 and score_midi[cur_pos]!= -1:
            no_move_flag = True
        # for stuck after long slience to wait for some sound
        elif pitch == -1 and score_midi[cur_pos+1]!= -1:
            no_move_flag = True
        else:
            no_move_flag = False

        plot = False
        plot_position = -1
        if plot and cur_pos > plot_position:
                start = max(cur_pos-100,0)
                end  = min(len(fsource),cur_pos+100)
                y1 = fsource[start:end]
                x = []
                y2 = f_I_given_D[start:end]
                y3 = f_V_given_I[start:end]
                y4 = score_midi[start:end]
                for k in range(start,end):
                    x.append(k*resolution)                
                fig=plt.figure()
                ax1 = plt.subplot(511)
                ax1.label_outer()
                plt.plot(x,y1)
                ax2 = plt.subplot(512,sharex=ax1)
                ax2.label_outer()
                plt.plot(x,y2, 'tab:orange')
                ax3 = plt.subplot(513,sharex=ax2)
                ax3.label_outer()
                plt.plot(x,y3, 'tab:green')
                ax4 = plt.subplot(514)
                plt.plot(x,y4, 'tab:red')
                plt.xlabel('Time in score {} pitch is {}'.format(cur_pos*resolution,pitch))
                plt.show()


        if fsource[cur_pos] > confidence[int(cur_time / resolution)]:
            confidence[int(cur_time / resolution)] = fsource[cur_pos]
        old_time = cur_time

        old_idx = onset_idx
        while onset_idx < len(onsets) and cur_pos >= onsets[onset_idx]:
            onset_idx += 1

        for i in range(len(onsets)):
            if cur_pos < onsets[i]:
                break
            old_idx = i - 1
        
        if confidence_record_check == 1:
            confidence_queue.append(confidence[int(cur_time / resolution)])
            confidence_record_check = 0
        # print("cur_time " + str(cur_time))
        # print("cur_midipitch" + str(score_midi[cur_pos]))
        # print "end_time %f" % (float(time.clock())-float(start_time))


# if __name__ == "__main__":
#     # start real time
#     # real_time = time.clock()
#     # score_follow(audio_file=AUDIOFILE, midi_file=MIDIFILE, feature='onset', mask=FILTER)
#     audio_end_time = 20
#     Rc = 67
#     resolution = 0.01
#     sQueue = []
#     audio_name = "audio2"
#     AUDIOFILE = 'audio/{}.wav'.format(audio_name)
#     midi_name = "midi2"
#     MIDIFILE = 'midi/{}.mid'.format(midi_name)
#     midi_file = MIDIFILE
#     score_midi, score_axis, score_onsets, onsets, raw_score_midi = get_time_axis(resolution,midi_file)
#     score_midi = score_midi_modification(score_midi)
#     # print(score_midi)
#     scoreLen = len(score_axis)
#     fsource = np.zeros(scoreLen)
#     confidence = np.zeros(scoreLen)
#     score_following(AUDIOFILE,audio_end_time,Rc,resolution,sQueue,fsource,scoreLen,confidence,score_midi)


# print(score_axis)
# def set_up_score_following(audio_file, midi_file, feature, mask):
