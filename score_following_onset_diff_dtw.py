#!/usr/bin/env python
# -*- coding: utf-8 -*-
import threading
import pretty_midi
import scikits.audiolab
import pyaudio
import analyse
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
import scipy.stats
import os
import librosa
import statsmodels.api as sm


std = 1 
resolution = 0.01
CHUNK = 1412
time_int = float(1412) / 44100
alpha = 10
score_start_time = 0


# vineyard
# Rc = 80
# zhui guang zhe
# Rc = 74
# shuo san jiu san
Rc = 70
#nanshannan
# Rc = 67
# yiqiannian don't know
# Rc = 74


# DIRECTORY = '/Users/rtchen/Downloads/sing_dataset/003/'
# DIRECTORY = '/Users/rtchen/Downloads/onset_detection/changba/0006 - 嘴巴嘟嘟/'
# AUDIOFILE = os.path.join(DIRECTORY,'audio.wav')
# MIDIFILE = os.path.join(DIRECTORY,'midi.mid')
# PITCHFILE = os.path.join(DIRECTORY,'pitch.txt')

plot = False
FILTER = 'gate'
gap_judge = False
name = "3"
AUDIOFILE = 'audio{}.wav'.format(name)
MIDIFILE = 'midi{}.mid'.format(name)
PITCHFILE = 'pitch3.txt'
NEWFILE = 'score_generated{}.mid'.format(name)
CONF_FILE = '{}filter_confidence.txt'.format(FILTER)
# Total time
total_time = 50

performance_start_time = 0
score_end_time = total_time
WINSIZE = 1
BREAK_TIME = total_time
WEIGHT = [0.5]
time_list_for_beat = [0]
se_list = [0]
confidence_queue = []
pos_list = [0]
pitch_pos_list = [0]



def normpdf(x, mean, sd):
    var = float(sd) ** 2
    pi = 3.1415926
    denom = (2 * pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom


def get_time_axis(resolution, start_time, end_time, filename):
    axis_start_time = 0
    axis_end_time = math.floor(end_time / resolution) * resolution
    size = (axis_end_time - axis_start_time) / resolution + 1
    axis = np.linspace(axis_start_time, axis_end_time, size)
    if abs(axis_end_time - end_time) > 1e-5:
        axis = np.concatenate((axis, [end_time]))
    scoreLen = len(axis)
    score_midi = np.zeros(scoreLen)
    midi_data = pretty_midi.PrettyMIDI(filename)
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
                score_midi[j] = note.pitch - note.pitch / 12 * 12
                # score_midi[j] = note.pitch
    # # to check
    # plt.plot(score_onsets)
    # plt.show()
    return score_midi, axis, score_onsets, onsets


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


def compute_f_V_given_I(pitch, pitches, scoreLen, score_midi, onset_prob, score_onsets, alpha, feature, w1, w2, w3):
    f_V_given_I = np.zeros(scoreLen)
    sims = np.zeros(scoreLen)

    if len(pitches) > WINSIZE:
        pitch = pitch - np.dot(pitches[-1 - WINSIZE:-1], WEIGHT)
    elif len(pitches) > 1:
        pitch = pitch - float(sum(pitches[:-1])) / (len(pitches) - 1)
    else:
        pitch = 0

    for i in range(scoreLen):

        if i >= WINSIZE:
            score_pitch = score_midi[i] - np.dot(score_midi[i - WINSIZE:i], WEIGHT)
        elif i > 0:
            score_pitch = score_midi[i] - float(sum(score_midi[:i])) / i
        else:
            score_pitch = 0

        score_onset = score_onsets[i]
        if feature == 'onset':
            # print "------------------onset pitch \n"
            # print "frist check"
            # print score_pitch
            # print pitch
            # print onset_prob
            # print score_onset
            if abs(pitch-score_pitch)<6:
                f_V_given_I[i] = math.pow(
                math.pow(normpdf(pitch, score_pitch, std), w1) * math.pow(similarity(onset_prob, score_onset), w2), w3)
            else:
            # f_V_given_I[i] = normpdf(pitch, score_pitch, std)
                f_V_given_I[i] =  similarity(onset_prob,score_onset)
        elif feature == 'uniform':
            f_V_given_I[i] = 1
        elif feature == 'only':
            f_V_given_I[i] = similarity(onset_prob, score_onset)
        elif feature == 'both':
            f_V_given_I[i] = math.pow(normpdf(pitch, score_pitch, std), 0.5) * math.pow(
                similarity(onset_prob, score_onset), 0.5)
        else:
            f_V_given_I[i] = normpdf(pitch, score_pitch, std)
        sims[i] = similarity(onset_prob, score_onset)

    # plt.plot(f_V_given_I)
    # plt.show()

    return f_V_given_I, sims


def compute_f_I_J_given_D(score_axis, estimated_tempo, elapsed_time, beta):
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
    # plt.plot(distribution)
    # plt.show()
    return distribution


def pitch_detection(data):
    samps = np.fromstring(data, dtype=np.int16)
    pitch = analyse.musical_detect_pitch(samps)
    if analyse.loudness(samps) > -25 and pitch != None:
        return pitch
    else:
        return -1


def tempo_estimate(elapsed_time, cur_pos, old_pos):
    # print 'cur pos %d old pos %d'%(cur_pos,old_pos)
    # print 'elapsed_time %f'%elapsed_time
    return float(cur_pos - old_pos) * Rc * resolution / elapsed_time


def compute_f_I_given_D(fsource, f_I_J_given_D, cur_pos, scoreLen):
    left = max(0, cur_pos - 1000)
    right = min(scoreLen, cur_pos + 1000)
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

    # f_I_given_D = np.convolve(fsource, f_I_J_given_D)
    return f_I_given_D


def plot_onsets_prob(onset_audio_file, scoreLen):
    # audio_onsets =plot_score(plot_audio_file = onset_audio_file)

    proc = CNNOnsetProcessor()
    audio_onsets = proc(onset_audio_file)
    # plt.plot(audio_onsets)
    # plt.show()
    return audio_onsets


def find_next_onset(cur_pos, score_midi, skip):
    while True:
        if cur_pos + skip > len(score_midi) - 1:
            cur_pos = len(score_midi) - 1
            break
        cur_pos = cur_pos + skip
        # print  'cur_pos  is  -------- %d'%cur_pos
        while score_midi[cur_pos] == 0 and cur_pos < len(score_midi) - 1:
            cur_pos += 1
        silence_cnt = 0
        total_cnt = 0
        for i in range(cur_pos, cur_pos + 200):
            if cur_pos < len(score_midi) and score_midi[cur_pos] == 0:
                silence_cnt += 1
            elif cur_pos < len(score_midi):
                total_cnt += 1
        if float(silence_cnt) / float(total_cnt + 1e-5) <= 0.8:
            break

    return cur_pos


def create_gaussian_mask(cur_pos, end_time, resolution):
    axis_end_time = math.floor(end_time / resolution) * resolution
    size = (axis_end_time - 0) / resolution + 1
    axis = np.linspace(0, axis_end_time, size)
    if abs(axis_end_time - end_time) > 1e-5:
        axis = np.concatenate((axis, [end_time]))

    mean = cur_pos * resolution
    std = 0.7
    gaussian_mask = scipy.stats.norm.pdf(axis, mean, std)

    # plt.plot(axis,gaussian_mask, color='black')
    # plt.show()
    return gaussian_mask


def create_gate_mask(cur_pos, scoreLen):
    mask = np.zeros(scoreLen)

    for i in range(-50, 51):
        if cur_pos + i < scoreLen and cur_pos + i >= 0:
            mask[cur_pos + i] = 1

    # plt.plot(range(scoreLen),mask, color='black')
    # plt.show()
    return mask


def score_follow(audio_file, midi_file, feature, mask):
    old_midi = pretty_midi.PrettyMIDI(MIDIFILE)
    new_midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    score_midi, score_axis, score_onsets, onsets = get_time_axis(resolution, score_start_time, score_end_time,
                                                                 midi_file)
    # score_midi = diff(score_midi)
    # plt.plot(score_midi)
    # plt.show()
    onset_idx = 0
    pitches = []
    mapped_onset_times = []
    mapped_onset_notes = []
    mapped_detect_notes = []
    scoreLen = len(score_axis)
    fsource = np.zeros(scoreLen)
    confidence = np.zeros(scoreLen)
    pitchfile = np.zeros(scoreLen)
    fsource[onsets[0]] = 1
    old_pos = 0
    cur_pos = 0
    tempo_estimate_elapsed_time = 0
    # for beat count
    tempo_estimate_elapsed_time2 = 0
    estimated_tempo = Rc
    matched_score = []
    detected_pitches = []
    time_axis = []
    mapped_time = []
    tempos = []
    wf = wave.open(AUDIOFILE, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    old_time = performance_start_time
    cur_time = performance_start_time
    firstReadingFlag = True
    n_frames = int(performance_start_time * wf.getframerate())
    wf.setpos(n_frames)
    data = wf.readframes(CHUNK)
    if feature == 'onset':
        audio_onsets = plot_onsets_prob(audio_file, scoreLen)
    else:
        audio_onsets = np.zeros(scoreLen)

    '''
    plt.plot(audio_onsets)
    plt.show()
    '''
    pitches = []
    last_silence_time = 0
    silence_cnt = 0
    datas = [data]
    confidence_record_check = 0
    raw_pitches = []
    while wf.tell() < wf.getnframes():
        if plot:
            stream.write(data)
        if len(datas) >= 3:
            c_data = datas[-3:]
            c_data = ''.join(c_data)
        else:
            c_data = data
        # c_data = data
        pitch = pitch_detection(c_data)
        if pitches == -1: 
           raw_pitches.append(0)
        else:
           raw_pitches.append(pitch)
        data = wf.readframes(CHUNK)
        datas.append(data)
        if firstReadingFlag:
            old_time = cur_time
        else:
            tempo_estimate_elapsed_time += time_int
            tempo_estimate_elapsed_time2 += time_int
        cur_time += time_int

        if cur_time > BREAK_TIME:
            break
        if gap_judge:
            if pitch == -1:
                if cur_time - last_silence_time <= time_int + 0.1 and not firstReadingFlag:
                    silence_cnt += 1
                    last_silence_time = cur_time
                elif not firstReadingFlag:
                    silence_cnt = 1
                    last_silence_time = cur_time
                if silence_cnt == 35:
                    fsource = np.zeros(scoreLen)
                    new_pos = find_next_onset(cur_pos, score_midi, 100)
                    print 'new pos %d' % new_pos
                    last_pos = cur_pos
                    old_pos = new_pos
                    cur_pos = new_pos
                    fsource[new_pos] = 1
                    timeQueue = []
                if silence_cnt >= 35:
                    old_time = cur_time
                    tempo_estimate_elapsed_time = 0
                    # tempo_estimate_elapsed_time2 = 0
                    # print 'old tempo %f'%tempo
                if silence_cnt == 100 and cur_pos - last_pos < 300:
                    fsource = np.zeros(scoreLen)
                    new_pos = find_next_onset(cur_pos, score_midi, 200)
                    print 'new pos ------%d' % new_pos
                    old_pos = new_pos
                    cur_pos = new_pos
                    fsource[new_pos] = 1
                if silence_cnt == 200 and cur_pos - last_pos < 600:
                    fsource = np.zeros(scoreLen)
                    new_pos = find_next_onset(cur_pos, score_midi, 300)
                    print 'new pos ------- %d' % new_pos
                    old_pos = new_pos
                    cur_pos = new_pos
                    fsource[new_pos] = 1
                continue

        pitch = pitch - int(pitch) / 12 * 12
        print 'detected pitch is %f' % pitch
        pitches.append(pitch)

        detected_pitches.append(-pitch)
        firstReadingFlag = False
        elapsed_time = cur_time - old_time
        tempo = estimated_tempo

        # to record the beat time
        if tempo * tempo_estimate_elapsed_time2 > 1 * Rc:
            print "bigger than one RC"
            confidence_record_check = 1
            pos_list.append(cur_pos)
            time_list_for_beat.append(time_list_for_beat[-1] + tempo_estimate_elapsed_time2)

        # change every two beats
        if tempo * tempo_estimate_elapsed_time > 2 * Rc:
            print "tempo changed"
            tempo = tempo_estimate(tempo_estimate_elapsed_time, cur_pos, old_pos)
            if tempo / float(Rc) < 0.9:
                tempo = Rc * 0.9
            elif tempo / float(Rc) > 1.1:
                tempo = Rc * 1.1
            tempo_estimate_elapsed_time = 0
            estimated_tempo = tempo
            old_pos = cur_pos

        print 'tempo %f' % tempo

        if int((cur_time) / 0.01) < len(audio_onsets):
            onset_prob = audio_onsets[int((cur_time) / 0.01)]
        else:
            onset_prob = 0

        if fsource[cur_pos] > 0.1:
            beta = 0.5
            w1 = 0.95
            w2 = 0.05
            w3 = 0.5
            #bigger w3 means pitch weight more
            # print '-----------------------------'
        else:
            beta = 0.5
            w1 = 0.7
            w2 = 0.3
            w3 = 0.3
            # w3 = 0.3



        f_I_J_given_D = compute_f_I_J_given_D(score_axis, tempo, elapsed_time, beta)
        f_I_given_D = compute_f_I_given_D(fsource, f_I_J_given_D, cur_pos, scoreLen)
        cur_pos = np.argmax(f_I_given_D)

        # F_V_give_I is the p(o|st)
        # F_I_give_D is the p(st)
        # f_source = p(st+1)

        f_V_given_I, sims = compute_f_V_given_I(pitch, pitches, scoreLen, score_midi, onset_prob, score_onsets, alpha,
                                                feature, w1, w2, w3)

        fsource = f_V_given_I * f_I_given_D[:scoreLen]
        if mask == 'gaussian':
            gaussian_mask = create_gaussian_mask(cur_pos, score_end_time, resolution)
            fsource = fsource * gaussian_mask

        elif mask == 'gate':
            gate_mask = create_gate_mask(cur_pos, scoreLen)
            fsource = fsource * gate_mask

        # fsource_real = fsource
        fsource = fsource / sum(fsource)
        cur_pos = np.argmax(fsource)
        # print "check -------"
        # print fsource[cur_pos]
        # print f_I_given_D[cur_pos]
        # print f_V_given_I[cur_pos]
        
        if plot:
            # if tempo * tempo_estimate_elapsed_time2 > 1 * Rc:
            # 100 curpose
                # print "pitch is "
                # print pitch
                start = max(cur_pos-100,0)
                end  = min(len(fsource),cur_pos+100)
                y1 = fsource[start:end]
                x = []
                y2 = f_I_given_D[start:end]
                y3 = f_V_given_I[start:end]
                y4 = score_midi[start:end]
                for k in range(start,end):
                    x.append(k*resolution)
                
                # , sharex = ax1, sharey = ax1
                fig=plt.figure()
                ax1 = plt.subplot(411)
                ax1.label_outer()
                plt.plot(x,y1)
                ax2 = plt.subplot(412,sharex=ax1)
                # ax2 = plt.subplot(412,sharex = ax1,sharey = ax1)
                ax2.label_outer()
                plt.plot(x,y2, 'tab:orange')
                ax3 = plt.subplot(413,sharex=ax2)
                # ax3 = plt.subplot(413,sharex = ax1,sharey = ax1)
                ax3.label_outer()
                plt.plot(x,y3, 'tab:green')
                ax4 = plt.subplot(414)
                plt.plot(x,y4, 'tab:red')
                # if len(pitches) > WINSIZE:
                #     pitch = pitch - np.dot(pitches[-1 - WINSIZE:-1], WEIGHT)
                # elif len(pitches) > 1:
                #      pitch = pitch - float(sum(pitches[:-1])) / (len(pitches) - 1)
                # else:
                #     pitch = 0
                plt.xlabel('Time in score {} pitch is {}'.format(cur_pos*resolution,pitch))
                plt.show()
                # name = ''
                # plt.savefig("check{}.png".format(cur_pos*resolution))
                # plt.clf() 

        if tempo * tempo_estimate_elapsed_time2 > 1 * Rc:
            pitch_pos_list.append(len(raw_pitches)-1)
            if len(raw_pitches)>=31 and cur_pos>=99 and len(pos_list) > 4:
                A_array = np.asmatrix(raw_pitches)
                B_array = np.asmatrix(score_midi[0:cur_pos+1])
                D, wp = librosa.sequence.dtw(A_array, B_array)
                for lst in wp:
                    if lst[0] == pitch_pos_list[-1]:
                        first_beat_back_pos = lst[1]
                    if lst[0] == pitch_pos_list[-2]:
                        second_beat_back_pos = lst[1]
                    if lst[0] == pitch_pos_list[-3]:
                        third_beat_back_pos = lst[1]
                    if lst[0] == pitch_pos_list[-4]:
                        forth_beat_back_pos = lst[1]
                    if lst[0] == pitch_pos_list[-5]:
                        fifth_beat_back_pos = lst[1]
                # alignCost= D[len_A-1,len_B-1]
                # we need time for each beat
                # we need the new position for each beat
                # we need to update the four beat's beat position
                # beat position is calculated by new tempo and elapsed_time
                # new tempo is calculated by elapsed time and new position
                # print time_list_for_beat
                # print pos_list
                # print pitch_pos_list
                tempo1 = tempo_estimate(time_list_for_beat[-1]-time_list_for_beat[-2],first_beat_back_pos,second_beat_back_pos)
                tempo2 = tempo_estimate(time_list_for_beat[-2]-time_list_for_beat[-3],second_beat_back_pos,third_beat_back_pos)
                tempo3 = tempo_estimate(time_list_for_beat[-3]-time_list_for_beat[-4],third_beat_back_pos,forth_beat_back_pos)
                tempo4 = tempo_estimate(time_list_for_beat[-4]-time_list_for_beat[-5],forth_beat_back_pos,fifth_beat_back_pos)

                # print tempo2
                if tempo1 / float(Rc) < 0.7:
                    tempo1 = Rc * 0.7
                elif tempo1 / float(Rc) > 1.3:
                    tempo1 = Rc * 1.3

                if tempo2 / float(Rc) < 0.7:
                    tempo2 = Rc * 0.7
                elif tempo2 / float(Rc) > 1.3:
                    tempo2 = Rc * 1.3

                if tempo3 / float(Rc) < 0.7:
                    tempo3 = Rc * 0.7
                elif tempo3 / float(Rc) > 1.3:
                    tempo3 = Rc * 1.3

                if tempo4 / float(Rc) < 0.7:
                    tempo4 = Rc * 0.7
                elif tempo4 / float(Rc) > 1.3:
                    tempo4 = Rc * 1.3



                beat1 = tempo1 * (time_list_for_beat[-1]-time_list_for_beat[-2])
                beat2 = tempo2 * (time_list_for_beat[-2]-time_list_for_beat[-3])+beat1
                beat3 = tempo3 * (time_list_for_beat[-3]-time_list_for_beat[-4])+beat2
                beat4 = tempo4 * (time_list_for_beat[-4]-time_list_for_beat[-5])+beat3

                x = time_list_for_beat[-5::]
                y = [0,beat1,beat2,beat3,beat4]
                # print "check --------------------x y"
                # print x
                # print y
                x = sm.add_constant(x)
                wls_model = sm.WLS(y, x)
                results = wls_model.fit()
                se = results.params[1]
                se_list.append(se/Rc)
                # print "se s ---------------------------------companre"
                # print se/Rc
                # print estimated_tempo
            else:
                se_list.append(0)
                
            tempo_estimate_elapsed_time2 = 0




        if fsource[cur_pos] > confidence[int(cur_time / resolution)]:
            confidence[int(cur_time / resolution)] = fsource[cur_pos]

        pitchfile[int(cur_time / resolution)] = pitch
        old_time = cur_time

        old_idx = onset_idx
        while onset_idx < len(onsets) and cur_pos >= onsets[onset_idx]:
            onset_idx += 1

        if old_idx < onset_idx:
            # for i in range(old_idx,onset_idx):
            old_note = old_midi.instruments[0].notes[onset_idx - 1]
            dur = old_note.end - old_note.start
            new_note = pretty_midi.Note(velocity=old_note.velocity, pitch=old_note.pitch, start=cur_time,
                                        end=cur_time + dur)
            piano.notes.append(new_note)

        for i in range(len(onsets)):
            if cur_pos < onsets[i]:
                break
            old_idx = i - 1

        if confidence_record_check == 1:
            confidence_queue.append(confidence[int(cur_time / resolution)])
            confidence_record_check = 0

        print 'currently at %d' % cur_pos
        print 'cur_time %f' % cur_time

     
        matched_score.append(score_midi[cur_pos])
        time_axis.append(cur_time - performance_start_time)
        mapped_time.append(cur_pos * resolution)
        tempos.append(tempo)

    new_midi.instruments.append(piano)
    new_midi.write(NEWFILE)
    np.savetxt(CONF_FILE, confidence)
    np.savetxt(PITCHFILE, pitchfile)
    score_plt = [min(-x / 5, 0) for x in score_midi]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    s, = plt.plot(score_plt, score_axis, color='r')
    dp, = plt.plot(time_axis, detected_pitches, 'o', color='g', label='detected pitch', markersize=1)
    mp, = plt.plot(time_axis, mapped_time, 'o', color='b', label='mapped position', markersize=1)
    plt.xlabel('audio time (seconds)')
    plt.ylabel('score time (seconds)')
    plt.grid()
    plt.show()



if __name__ == "__main__":
    score_follow(audio_file=AUDIOFILE, midi_file=MIDIFILE, feature='onset', mask=FILTER)
    print "Beat list ----------------------"
    print time_list_for_beat
    print "se_list -------------------------"
    print se_list
    print "confidence_list -------------------------"
    print confidence_queue
    time_beat_name = "time_beat_file{}.txt".format(name)
    confidence_name = "confidence_queue_file{}.txt".format(name)
    se_list_name = "se_list_file{}.txt".format(name)
    time_beat_file = open(time_beat_name, 'w')
    confidence_queue_file = open(confidence_name, 'w')
    se_list_file = open(se_list_name,'w')
    for item in time_list_for_beat:
        time_beat_file.write(str(item))
        time_beat_file.write("\n")
    for item2 in confidence_queue:
        confidence_queue_file.write(str(item2))
        confidence_queue_file.write("\n")
    for item3 in se_list:
        se_list_file.write(str(item3))
        se_list_file.write("\n")

