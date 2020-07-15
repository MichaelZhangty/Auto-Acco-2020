#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import time
import threading
import fluidsynth
from scipy import stats
from auto_acco_combine_utilities import *

# BPM parameter for each midi
# zhui guang zhe
# BPM = 74
# shuo san jiu san
# BPM = 70
# nanshannan
# BPM = 67
BPM = 74
Rc = 74

# set up all the parameter first 
# refine score_following 
# put each time_list_for_beat and confidence 
# stop every 0.023s and provide 0.023s more 
# first version is for muti-thread to create auto-acco


# auto accompany paramters 
audio_name = "audio2"
midi_name = "midi2"
end_time = 20
audio_end_time = 20
confidence_queue = [0.001, 0.001, 0.001, 0.001, 0.001]
BPS = BPM / float(60)  # beat per second
original_begin = time.clock()
global_tempo = 0
# weight is for 0.5beats for 2 beats
weight_judge = True
beat_back = 4
pressed_key = "lol"
stop_thread = False
sQueue = []
latency_end = -1
resolution = 0.01
# score following parameters
AUDIOFILE = 'audio/{}.wav'.format(audio_name)
MIDIFILE = 'midi/{}.mid'.format(midi_name)
ACC_FILE = 'midi/{}.mid'.format(midi_name)
# set up for score following
sQueue = []
midi_file = MIDIFILE
score_midi, score_axis, score_onsets, onsets, raw_score_midi = get_time_axis(resolution,midi_file)
score_midi = score_midi_modification(score_midi)
# print(score_midi)
scoreLen = len(score_axis)
fsource = np.zeros(scoreLen)
confidence = np.zeros(scoreLen)
# thread for simulating the audio file
def press_key_thread():
    global pressed_key
    global stop_thread
    global latency_end
    while not stop_thread:
        # use score_following to record the timeQueue and confidence_queue
        score_following(AUDIOFILE,audio_end_time,Rc,resolution,sQueue,fsource,scoreLen,confidence,score_midi,latency_end)


'''
function: compute_tempo_ratio:
----------------------------------------------------
scheduleing algo by NIME 2011 paper
'''


old_midi = pretty_midi.PrettyMIDI(ACC_FILE)
new_midi = pretty_midi.PrettyMIDI()
piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
piano = pretty_midi.Instrument(program=piano_program)


class Player:
    def __init__(self, ACC_FILE, original_begin, BPM, fs):
        self.ACC_FILE = ACC_FILE
        self.original_begin = original_begin
        self.midi_data = pretty_midi.PrettyMIDI(ACC_FILE)
        self.notes = sorted(self.midi_data.instruments[0].notes, key=lambda x: x.start, reverse=False)
        self.resolution = 0.01
        self.start_time = 0
        self.end_time = end_time
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
        global time_list_beat
        global latency_end
        begin = time.clock()
        total_delay = 0

        for i in range(start, len(self.notes)):
            note = self.notes[i]
            cur_time = time.clock() - begin - total_delay
            wait_delta = note.start - cur_time
            # cur=49
            if cur_time > end_time:
                break

            tempo_ratio = float(self.BPS) / sQueue[-1]
            # print sQueue
            print "Tempo_ratio == %f" % tempo_ratio
            total_delay += wait_delta * (tempo_ratio-1)
            wait_delta = wait_delta * tempo_ratio


            target_start_time = time.clock() + wait_delta
            latency_end = target_start_time
            while time.clock() < target_start_time:
                pass
            # print "--------------------------- wrong %f" %time.clock()
            self.playTimes.append(time.clock() - original_begin)
            self.noteTimes.append(note.start)
            tempo_ratio = float(self.BPS) / sQueue[-1]
            cur_time = time.clock() - begin - total_delay
            new_note = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=cur_time, end=cur_time + note.end - note.start)
            piano.notes.append(new_note)
            delta_time = note.end - (time.clock() - begin - total_delay)
            # print "---------------------------correct %f" %time.clock()
            total_delay += delta_time * (tempo_ratio - 1)
            delta_time = delta_time * tempo_ratio
            target_time = time.clock() + delta_time
            latency_end = target_time
            while time.clock() < target_time:
                pass
            # self.fs.noteoff(0, note.pitch)
            # for count for long break
            old_target_start_time = target_start_time + 0

        tap_time = [t for t in time_list_beat]
        tap_beat = [(t - 4) / float(self.BPS) for t in range(len(tap_time))]
        new_midi.instruments.append(piano)
        new_midi.write("auto_accompany/auto_accompany{}.mid".format(audio_name))
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
            if len(time_list_beat) >= 5:
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
