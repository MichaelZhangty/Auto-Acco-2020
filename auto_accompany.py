import pretty_midi
import time
import threading, time
import matplotlib.pyplot as plt
import fluidsynth
# print(dir(fluidsynth))
# pyFluidSynth	1.2.5	1.2.5
import sys
from scipy import stats

# next week
# 1 change beats by 0.5 or less, look back 2 beats
# 2 diff and 1-x pitch
# 3 check back several seconds for onset
# 4 how to interact score following and vocal tracking

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
# import librosa
# from google.cloud import speech
# from google.cloud.speech import enums
# from google.cloud.speech import types
# import io
import scipy.stats
import os
import statsmodels.api as sm



# name of the song!!!
name = "3"


# file is for no gap
# # file1 is for with gap
time_beat_file = 'time_beat_file{}.txt'.format(name)
confidence_file = 'confidence_queue_file{}.txt'.format(name)
# for dtw
# time_beat_file = 'time_beat_file_DTW{}.txt'.format(name)
# confidence_file = 'confidence_queue_file_DTW{}.txt'.format(name)

# ACC_FILE = 'midi{}.mid'.format(name)
ACC_FILE = 'midi{}.mid'.format(name)



time_beat_file = open(time_beat_file, 'r')
confidence_queue_file = open(confidence_file, 'r')
beat_lines = time_beat_file.readlines()
confidence_lines = confidence_queue_file.readlines()
time_list_for_beat = []
confidence_queue = [0.001, 0.001, 0.001, 0.001, 0.001]
for item in beat_lines:
    time_list_for_beat.append(float(item))
for item2 in confidence_lines:
    confidence_queue.append(float(item2))
time_list_for_beat.pop(0)




# ACC_FILE = 'scale.mid'
# shuo san jiu san
Rc = 70
# vineyard
# Rc = 80
# # zhui guang zhe
# Rc = 74
BPM = 60
# BPM = 74
BPS = BPM / float(60)  # beat per second
original_begin = time.clock()
global_tempo = 0
# fluidsynth.init("soundfont.sf2")
# fluidsynth need some time to load sound file otherwise there is a
# chance there is no sound
# tmp  = raw_input('press return key to begin this program')

# weight is for 0.5beats for 2 beats
weight_judge = True
beat_back = 4
pressed_key = "lol"
timeQueue = []
stop_thread = False
sQueue = []
latency_end = -1
# for simulation
# speed = 0.8
# timeQueue = [-4,-3,-2,-1,0]
# x = timeQueue[-4:]
# y = range(1, 5)
# s0, intercept, r_value, p_value, std_err = stats.linregress(x, y)
# sQueue.append(s0)
# simu 2
# simulation_times = [1, 2, 3, 4, 5, 6, 7, 8]
simulation_times = [1, 2, 3, 4, 5]
for i in time_list_for_beat:
    simulation_times.append(i + 5)
# simulation_times = time_list_for_beat
# tmp = 8
# for i in range(28):
#     tmp = tmp + 2
#     simulation_times.append(tmp)

'''
function: press_key_thread:
--------------------------------------------------------
init another thread, take in keyboard input
each tap on return/enter key is registered as a beat
and save to global queue timeQueue
'''
from playsound import playsound


def press_key_thread():
    global pressed_key
    global stop_thread
    global latency_end
    cnt = 0
    while not stop_thread:
        # pressed_key = sys.stdin.readline()
        # if pressed_key=='\n':
        # origianl version
        # if cnt < len(simulation_times) and (
        #         abs(time.clock() - original_begin - simulation_times[cnt])) <= 0.003 or time.clock() - original_begin >= \
        #         simulation_times[cnt]:
        # new version:
        if cnt < len(simulation_times) and (
                 abs(time.clock() - original_begin - simulation_times[cnt])) <= 0.003 or time.clock() - original_begin >= \
                 simulation_times[cnt]:
            timeQueue.append(time.clock() - original_begin)
            cnt = cnt + 1
            # print timeQueue
            if len(timeQueue) == 5:
                print "timeQueue == 5"
                if weight_judge:
                    b0 = 1
                    t0 = timeQueue[-1]
                    x = timeQueue[-beat_back:]
                    y = range(1, beat_back + 1)
                    # print x
                    # print y
                    confidence_block = confidence_queue[len(timeQueue) - beat_back:len(timeQueue) - 0]
                    sum_confidence = sum(confidence_block)
                    confidence_count = []
                    for index in range(len(confidence_block)):
                        confidence_count.append(round(confidence_block[index] / sum_confidence, 1))
                    # print confidence_count
                    for index in range(len(confidence_count)):
                        print confidence_count[index]
                        times = int(confidence_count[index] * 10)
                        for i in range(times - 1):
                            x.append(x[index])
                            y.append(y[index])
                    # print x
                    # print y
                    s0, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    sQueue.append(s0)
                else:
                    b0 = 1
                    t0 = timeQueue[-1]
                    s0 = float(1) / (timeQueue[-2] - timeQueue[-3])
                    x = timeQueue[-4:]
                    y = range(1, 5)
                    # s0, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                    sQueue.append(s0)
            if len(timeQueue) % 2 == 1 and len(timeQueue) > 5:
                if latency_end == -1:
                    l = 0.1
                else:
                    l = max(0, latency_end - time.clock())
                if weight_judge:
                    b0, t0, s0 = compute_tempo_ratio_weighted(b0, t0, s0, l)
                else:
                    b0, t0, s0 = compute_tempo_ratio(b0, t0, s0, l)
            # extra
            # if len(timeQueue) % 2 == 0 and len(timeQueue) > 5:
            #     if latency_end == -1:
            #         l = 0.1
            #     else:
            #         l = max(0, latency_end - time.clock())
            #     if weight_judge:
            #         b0, t0, s0 = compute_tempo_ratio_weighted(b0, t0, s0, l)
            #     else:
            #         b0, t0, s0 = compute_tempo_ratio(b0, t0, s0, l)
            pressed_key = 'lol'


'''
function: compute_tempo_ratio:
----------------------------------------------------
scheduleing algo by NIME 2011 paper
'''

# for dtw to compute the axis


def compute_tempo_ratio(b0, t0, s0, l):
    t1 = timeQueue[-1]
    b = b0 + (t1 + l - t0) * s0
    tn = t1 + l
    bn = b
    te = timeQueue[-2]
    # be = len(timeQueue) - 5
    be = len(timeQueue) - 5
    se = float(1) / (timeQueue[-1] - timeQueue[-2])
    # for normal regression
    sn = (float(4) / (te * se - tn * se - be + bn + 4)) * se
    # sn = (float(4) * se / (te * se - tn * se - be + bn + 4))
    # print(sn)
    sQueue.append(sn)
    return bn, tn, sn


def compute_tempo_ratio_weighted(b0, t0, s0, l):
    t1 = timeQueue[-1]
    b = b0 + (t1 + l - t0) * s0
    tn = t1 + l
    bn = b
    te = timeQueue[-2]
    # se = float(1) / (timeQueue[-1] - timeQueue[-2])
    # for long break to stop regression
    # global_tempo = float(1) / (timeQueue[-1] - timeQueue[-2])
    be = len(timeQueue) - 5
    x = timeQueue[-beat_back:]
    y = range(len(timeQueue) - beat_back - 3, len(timeQueue) - 3)
    confidence_block = confidence_queue[len(timeQueue) - beat_back:len(timeQueue) - 0]
    # print x
    # print y
    # print confidence_block
    # naive version
    # sum_confidence = sum(confidence_block)
    # confidence_count = []
    # for index in range(len(confidence_block)):
    #     confidence_count.append(round(confidence_block[index] / sum_confidence, 1))
    # # print confidence_count
    # for index in range(len(confidence_count)):
    #     # print confidence_count[index]
    #     times = int(confidence_count[index] * 10)
    #     for i in range(times - 1):
    #         x.append(x[index])
    #         y.append(y[index])
    # se, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # print "after weight"
    # # # print confidence_count
    # true weighted
    # print "------------------------------x timeQueue y range-----------------------"
    # print x
    # print y
    # print "----------------------------confidence_block"
    # print confidence_block
    x = sm.add_constant(x)
    if y[0] == 0:
        print "first one -------------------------------------"
        wls_model = sm.WLS(y, x)
        results = wls_model.fit()
        se = results.params[1]
    else:
        wls_model = sm.WLS(y, x, weights=confidence_block)
        results = wls_model.fit()
        se = results.params[1]
    print "------------------------------sk-----------------------"
    print se

    # weird version
    # print "------------------------------x timeQueue y range-----------------------"
    # print x
    # print y
    # # # x = sm.add_constant(x)
    # print "----------------------------confidence_block"
    # print confidence_block
    # # x = sm.add_constant(x)
    # if y[0] == 0:
    #     print "first one -------------------------------------"
    #     wls_model = sm.WLS(y, x)
    #     results = wls_model.fit()
    #     se = results.params[0]
    # else:
    #     wls_model = sm.WLS(y, x, weights=confidence_block)
    #     results = wls_model.fit()
    #     se = results.params[0]
    # print "------------------------------sk-----------------------"
    # print se


    # for normal regression
    sn = (float(4) / (te * se - tn * se - be + bn + 4)) * se
    # sn = (float(4) * se / (te * se - tn * se - be + bn + 4))
    # print(sn)
    sQueue.append(sn)
    return bn, tn, sn


old_midi = pretty_midi.PrettyMIDI(ACC_FILE)
new_midi = pretty_midi.PrettyMIDI()
piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
piano = pretty_midi.Instrument(program=piano_program)

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
    # # to check
    # plt.plot(score_onsets)
    # plt.show()
    return score_midi, axis, score_onsets, onsets


class Player:
    def __init__(self, ACC_FILE, original_begin, BPM, fs):
        self.ACC_FILE = ACC_FILE
        self.original_begin = original_begin
        self.midi_data = pretty_midi.PrettyMIDI(ACC_FILE)
        self.notes = sorted(self.midi_data.instruments[0].notes, key=lambda x: x.start, reverse=False)
        self.resolution = 0.01
        self.start_time = 0
        self.end_time = 50
        self.score_midi, self.axis, self.score_onsets, self.onsets = get_time_axis(self.resolution, self.start_time, self.end_time, self.ACC_FILE)
        self.playTimes = []
        self.noteTimes = []
        self.midi_start_time = self.notes[0].start
        # self.BPS = BPM / float(60)
        self.BPS = BPM / float(60)
        self.fs = fs

    '''
    function: follow:
    -----------------------------------------------
    follow score from start point
    '''

    def follow(self, start):
        global sQueue
        global timeQueue
        global latency_end
        begin = time.clock()

        for i in range(start, len(self.notes)):
            # for simulation
            # if len(timeQueue) == 5:
            #     b0 = 1
            #     t0 = timeQueue[-1]
            #     s0 = float(1)/(timeQueue[-2]-timeQueue[-3])
            #     x = timeQueue[-4:]
            #     y = range(1,5)
            #     # s0, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            #     sQueue.append(s0)
            # if len(timeQueue)%2==1 and len(timeQueue)>5:
            #     print(latency_end)
            #     if latency_end == -1:
            #        l = 0.1
            #     else:
            #        l = max(0,latency_end - time.clock())
            #     b0,t0,s0=compute_tempo_ratio(b0,t0,s0,l)
            # if len(timeQueue)%2==0 and len(timeQueue)>5:
            #     print(latency_end)
            #     if latency_end == -1:
            #        l = 0.1
            #     else:
            #        l = max(0,latency_end - time.clock())
            #     b0,t0,s0=compute_tempo_ratio(b0,t0,s0,l)

            note = self.notes[i]
            cur_time = time.clock() - begin
            wait_delta = note.start - cur_time
            # cur=49
            if cur_time > 49:
                break
            tempo_ratio = float(self.BPS) / sQueue[-1]
            if tempo_ratio < 1:
                begin -= wait_delta * (1 - tempo_ratio)
                wait_delta = wait_delta * tempo_ratio
            elif tempo_ratio > 1:
                begin += wait_delta * (tempo_ratio - 1)
                wait_delta = wait_delta * tempo_ratio

            target_start_time = time.clock() + wait_delta
            latency_end = target_start_time
            while time.clock() < target_start_time:
                # if time.clock() - old_target_start_time > 4 * (1 / tempo_ratio):
                #     tempo_ratio = global_tempo
                #     sQueue[-1] = global_tempo
                #     break
                pass

            delta_time = note.end - (time.clock() - begin)
            # print 'delta_time%f'%(delta_time-note.end+note.start)

            self.playTimes.append(time.clock() - original_begin)
            self.noteTimes.append(note.start)

            tempo_ratio = float(self.BPS) / sQueue[-1]
            if tempo_ratio < 1:
                print('tempo faster with ratio %f' % tempo_ratio)
                begin -= delta_time * (1 - tempo_ratio)
                delta_time = delta_time * tempo_ratio
            elif tempo_ratio >= 1:
                print('tempo slower with ratio %f' % tempo_ratio)
                begin += delta_time * (tempo_ratio - 1)
                delta_time = delta_time * tempo_ratio

            # old_note = self.midi_data.instruments[0].notes[i]
            # dur = old_note.end-old_note.start
            # new_note = pretty_midi.Note(velocity=old_note.velocity, pitch=old_note.pitch, start=old_note.start, end=old_note.end)
            # piano.notes.append(new_note)

            # normal version

            cur_time = time.clock() - begin
            dur = note.end - note.start
            # new version
            # dur = (note.end - note.start)*tempo_ratio
            new_note = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=cur_time, end=cur_time + dur)
            piano.notes.append(new_note)

            # self.fs.noteon(0, note.pitch, 100)
            target_time = time.clock() + delta_time
            latency_end = target_time
            while time.clock() < target_time:
                pass
            # self.fs.noteoff(0, note.pitch)

            # for count for long break
            old_target_start_time = target_start_time + 0
            # for simulation
            # new_gap = speed * (timeQueue[-1] - timeQueue[-2])
            # new_gap = 0.5
            # timeQueue.append(timeQueue[-1]+new_gap)
            # print(sQueue)
            # print(timeQueue)

        tap_time = [t for t in timeQueue]
        tap_beat = [(t - 4) / float(self.BPS) for t in range(len(tap_time))]
        # for simulation
        # tap_beat = [(t) / float(self.BPS) for t in range(len(tap_time))]
        # print(tap_time)
        # print(self.playTimes)
        # print(tap_beat)
        # print(self.noteTimes)
        # tap_beat = tap_beat[3::]
        # tap_time = tap_time[3::]
        # print(tap_beat)

        new_midi.instruments.append(piano)
        new_midi.write("auto_accompany{}.mid".format(name))

        plt.scatter(x=self.playTimes, y=self.noteTimes, c='b', s=10, marker='o')
        plt.plot(tap_time, tap_beat, marker='+')
        plt.xlabel('audio time (seconds)')
        plt.ylabel('score time (seconds)')
        plt.show()

    '''
    function:jump
    ---------------------------------------
    jump to specified ith note
    '''

    def jump(self, i):
        self.follow(i)


#
if __name__ == '__main__':
    pk_thread = threading.Thread(target=press_key_thread)
    pk_thread.start()
    fs = fluidsynth.Synth()
    sfid = fs.sfload("soundfont.sf2")
    fs.start("coreaudio")
    fs.program_select(0, sfid, 0, 0)
    try:
        # print('tap four times to start')
        while True:
            if len(timeQueue) >= 5:
                break
        player = Player(ACC_FILE, original_begin, BPM, fs)
        player.follow(0)
    except KeyboardInterrupt:
        stop_thread = True
        pk_thread.killed = True
        pk_thread.join()
        fs.delete()
    finally:
        stop_thread = True
        pk_thread.killed = True
        pk_thread.join()
        fs.delete()
