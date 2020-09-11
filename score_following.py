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
from score_following_utilities import *
import aubio
# Rc parameter for each midi
# vineyard
# Rc = 80
# zhui guang zhe 2
# Rc = 74
# shuo san jiu san 3
# Rc = 70
# nanshannan 4
# Rc = 67
# "shuosanjiusan_gus_6_9"
# parameter to change
audio_name = "audio3"
midi_name  = "midi3"
Rc = 70
score_end_time = 500
audio_end_time = 500



performance_start_time = 0
score_start_time = 0
# CHUNK = 1412
CHUNK = 1024
# CHUNK = 2048
resolution = 0.01  #float(CHUNK) / 44100 #0.01
time_int = float(CHUNK) / 44100
alpha = 10

plot = False
FILTER = 'gate'
aubio_pitch = True
aubio_onset = True
plot_position = 0
onset_help = False


time_list_for_beat = [0]
confidence_queue = []
pitch_pos_list = [0]
# tempo ratio bound
temp_upbound = 1.3
temp_downbound = 0.7

# for debug 
# tag_name = "_onset_judge"
# audio_name_save = audio_name + tag_name
audio_name_save = audio_name

# file to open and create 
AUDIOFILE = 'audio/{}.wav'.format(audio_name)
MIDIFILE = 'midi/{}.mid'.format(midi_name)
PITCHFILE = 'pitch/pitch_{}.txt'.format(audio_name_save)
NEWFILE = 'score_following/score_generated_{}.mid'.format(audio_name_save)
# CONF_FILE = 'confidence/{}_confidence.txt'.format(audio_name_save)
time_beat_name = "time_beat/time_beat_{}.txt".format(audio_name_save)
confidence_name = "confidence/confidence_queue_{}.txt".format(audio_name_save)


def score_follow(audio_file, midi_file, feature, mask):
    old_midi = pretty_midi.PrettyMIDI(MIDIFILE)
    # get midi tempo
    # midi_tempo = old_midi.estimate_tempo()
    new_midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    score_midi, score_axis, score_onsets, onsets, raw_score_midi,axis_loudness = get_time_axis(resolution, score_start_time, score_end_time,
                                                                 midi_file)
    onset_idx = 0
    pitches = []
    mapped_onset_times = []
    mapped_onset_notes = []
    mapped_detect_notes = []
    scoreLen = len(score_axis)
    fsource = np.zeros(scoreLen)
    confidence = np.zeros(scoreLen)
    pitchfile = np.zeros(scoreLen)
    # start from the beginning instead of first onset
    fsource[0] = 1
    # for check no movement
    last_cur_pos = 0
    no_move_flag = False
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
    last_beat_position_1 = 0 
    last_beat_position_2 = 0

    # get audio onset 
    # if feature == 'onset':
    #     if aubio_onset:
    #         audio_onsets = plot_onsets_prob_aubio(audio_file,scoreLen)
    #     else:
    #         audio_onsets = plot_onsets_prob(audio_file, scoreLen)
    # else:
    #     audio_onsets = np.zeros(scoreLen)

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

    # change score_midi to make every pitch last for at most 1.5s
    temp_pitch = 0
    count = 1
    onset_detector = aubio.onset("default", CHUNK, CHUNK, 44100)
    # for i in range(len(score_midi)):
    #     if score_midi[i] == temp_pitch:
    #         count += 1
    #         if count >= 100:
    #             score_midi[i] = -1
    #     else:
    #         temp_pitch = score_midi[i]
    #         count = 1
    #44100 
    #1412 0.03s
    #0.09s
    start_time = time.clock()
    count_cut = 1 
    count_jump = 0
    while wf.tell() < wf.getnframes():
        while time.clock() - start_time < count_cut * time_int:
            pass
        count_cut += 1
        # start_time = time.clock()
        # print("begin_time" + str(time.clock()))
        # play the frame
        # stream.write(data)
        if len(datas) >= 3:
            c_data = datas[-3:]
            c_data = b''.join(c_data)
        else:
            c_data = data
        if plot and cur_pos > plot_position:
            stream.write(data)

        # detect pitch 
        if aubio_pitch:
            if len(datas) >= 3:
                pitch = pitch_detection_aubio(c_data,3,CHUNK)
            else:
                pitch = pitch_detection_aubio(c_data,1,CHUNK)
            #for single chunk judge
            # c_data = data
            # pitch = pitch_detection_aubio(c_data,1,CHUNK)
        else:
            pitch = pitch_detection(c_data)


        if pitch == -1: 
           raw_pitches.append(-1)
        else:
           raw_pitches.append(pitch)
        
        # print("pitch_detected"+str(time.clock()))

        data_onset = np.fromstring(data, dtype=np.int16)
        data_onset = np.true_divide(data_onset, 32768, dtype=np.float32)
        # print("lengtjh-----"+str(len(data_onset)))
        if len(data_onset) == CHUNK: # 1024
            onset_detector(data_onset)
            try:
                onset_prob = onset_detector.get_last() / 44100
            except:
                onset_prob = 0

        data = wf.readframes(CHUNK)
        datas.append(data)

        if firstReadingFlag:
            old_time = cur_time
        else:
            tempo_estimate_elapsed_time += time_int
            tempo_estimate_elapsed_time2 += time_int
        cur_time += time_int

        if cur_time > audio_end_time:
            break

        # print("before detected pitch is " + str(pitch))
        # fix no sound bug
        # remove module 12 problem
        if pitch == -1:
            pitch = -1
        else:
            pitch = pitch%12
        # print 'detected pitch is %f' % pitch
        print("detected pitch is" + str(pitch))
        pitches.append(pitch)

        detected_pitches.append(-pitch)
        firstReadingFlag = False
        elapsed_time = cur_time - old_time
        # print("cur_time is" + str(cur_time))
        tempo = estimated_tempo

        # # record the beat time
        # if tempo * tempo_estimate_elapsed_time2 > 1 * Rc:
        #     # print "bigger than one RC"
        #     confidence_record_check = 1
        #     # pos_list.append(cur_pos) # only use for dtw
        #     # tempo_estimate_elapsed_time2 is performance time
        #     # print("record--------" + str(time_list_for_beat[-1] + tempo_estimate_elapsed_time2))
        #     time_list_for_beat.append(time_list_for_beat[-1] + tempo_estimate_elapsed_time2)
        #     tempo_estimate_elapsed_time2 = 0

        # # tempo_follow change every two beats
        # if tempo * tempo_estimate_elapsed_time > 2 * Rc:
        #     # print "tempo changed"
        #     tempo = tempo_estimate(tempo_estimate_elapsed_time, cur_pos, old_pos,Rc)
        #     # with bound for tempo
        #     if tempo / float(Rc) < temp_downbound:
        #         tempo = Rc * temp_downbound
        #     elif tempo / float(Rc) > temp_upbound:
        #         tempo = Rc * temp_upbound
        #     tempo_estimate_elapsed_time = 0
        #     estimated_tempo = tempo
        #     old_pos = cur_pos

        # print 'tempo %f' % tempo
        # print("tempo"+str(tempo))
        # print("current pos" + str(cur_pos))


        

        # if int((cur_time) / 0.01) < len(audio_onsets):
        #     onset_prob = audio_onsets[int((cur_time) / 0.01)]
        # else:
        #     onset_prob = 0

        # confidence > 0.1, pitch weight more 
        # orgianl set w1 = 0.95 w2 = 0.05 w3 = 0.5
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


        # print("after figivend cur pos is"+str(cur_pos))
        # print("set up before calculate"+str(time.clock()))
        fsource_original = fsource

        if no_move_flag:
            f_I_given_D = fsource
        else:
            f_I_J_given_D = compute_f_I_J_given_D(score_axis, tempo, elapsed_time, beta,alpha,Rc,no_move_flag)
            # print("compute delta" + str(time.clock()))
            f_I_given_D = compute_f_I_given_D(fsource, f_I_J_given_D, cur_pos, scoreLen)
        cur_pos = np.argmax(f_I_given_D)
        # print("move delta for p(ot)"+str(time.clock()))
        # print("after figivend cur pos is"+str(cur_pos))

        # F_V_give_I is the p(o|st)
        # F_I_give_D is the p(st)
        # f_source = p(st+1)

        f_V_given_I, sims = compute_f_V_given_I(pitch, pitches, scoreLen, score_midi, onset_prob, score_onsets, alpha,
                                                feature, w1, w2, w3,cur_pos)
        # print("compute observation" + str(time.clock()))

        # fsource = f_V_given_I * f_I_given_D[:scoreLen]
        fsource = f_V_given_I * f_I_given_D
        # print("compute po" + str(time.clock()))

        if mask == 'gaussian':
            gaussian_mask = create_gaussian_mask(cur_pos, score_end_time, resolution)
            fsource = fsource * gaussian_mask

        elif mask == 'gate':
            gate_mask = create_gate_mask(cur_pos, scoreLen)
            fsource = fsource * gate_mask
        # print(fsource)
        # print("fsource"+str(sum(fsource)))
        fsource = fsource / sum(fsource)
        cur_pos = np.argmax(fsource)
        # print("compute final position" + str(time.clock()))
        # if cur pitch is not onset
        # It go at most the position before next onset
        # print("onset_prob: "+str(onset_prob))
        # print("onset_idx: " + str(onsets[onset_idx]))

        # onset < 0.01

        # if onset_help and onset_prob < 0.01 and cur_pos >= onsets[onset_idx]:
        #     cur_pos = last_cur_pos
        #     fsource = fsource_last
        #     # print("help onset judge --------")
        #     # cur_pos = onsets[onset_idx] - 1
        # else:
        #     last_cur_pos = cur_pos
        #     fsource_last = fsource


        # for suddenly slient while singing in the middle       
        if pitch == -1 and score_midi[cur_pos]!= -1:
            no_move_flag = True
        # for stuck after long slience to wait for some sound
        if pitch == -1 and score_midi[cur_pos+1]!= -1:
            no_move_flag = True
        else:
            no_move_flag = False

        # if pitch != -1 and score_midi[cur_pos] == -1:
        #     count_jump += 1
        # else:
        #     count_jump = 0

        # if pitch > -1 and (axis_loudness[cur_pos] + axis_loudness[cur_pos + 1] == 2):
        #         no_move_flag = False
        # else:
        #         no_move_flag = True

        # if no_move_flag:
        #     print("no move --------------------------")
        #     fsource = fsource_original
        #     cur_pos = np.argmax(fsource)
        # last_cur_pos = cur_pos
        # fsource_last = fsource
        # print "check -------"
        # print fsource[cur_pos]
        # print f_I_given_D[cur_pos]
        # print f_V_given_I[cur_pos]
        
        if plot and cur_pos > plot_position:
                start = max(cur_pos-10,0)
                end  = min(len(fsource),cur_pos+10)
                y1 = fsource[start:end]
                x = []
                y2 = f_I_given_D[start:end]
                y3 = f_V_given_I[start:end]
                y4 = score_midi[start:end]
                for k in range(start,end):
                    x.append(k*resolution)                
                fig= plt.figure()
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


        # if fsource[cur_pos] > confidence[int(cur_time / resolution)]:
        #     confidence[int(cur_time / resolution)] = fsource[cur_pos]
        if fsource[cur_pos] > confidence[cur_pos]:
            confidence[cur_pos] = fsource[cur_pos]

        # pitchfile[int(cur_time / resolution)] = pitch
        pitchfile[cur_pos] = pitch
        old_time = cur_time

        old_idx = onset_idx
        while onset_idx < len(onsets) and cur_pos >= onsets[onset_idx]:
            onset_idx += 1

        if old_idx < onset_idx and not no_move_flag:
            print("append new note-----------------------------------------")
            old_note = old_midi.instruments[0].notes[onset_idx - 1]
            dur = old_note.end - old_note.start
            new_note = pretty_midi.Note(velocity=old_note.velocity, pitch=old_note.pitch, start=cur_time,
                                        end=cur_time + min(dur,1.5))
            piano.notes.append(new_note)
        old_idx = onset_idx - 1

        # for i in range(len(onsets)):
        #     if cur_pos < onsets[i]:
        #         break
        #     old_idx = i - 1

        
        # record the beat time
        # if tempo * tempo_estimate_elapsed_time2 > 1 * Rc:
        if (cur_pos-last_beat_position_1)/100 > 60/Rc:
            # print "bigger than one RC"
            confidence_record_check = 1
            # pos_list.append(cur_pos) # only use for dtw
            # tempo_estimate_elapsed_time2 is performance time
            # print("record--------" + str(time_list_for_beat[-1] + tempo_estimate_elapsed_time2))
            time_list_for_beat.append(time_list_for_beat[-1] + tempo_estimate_elapsed_time2)
            tempo_estimate_elapsed_time2 = 0
            last_beat_position_1 = cur_pos

        # tempo_follow change every two beats
        # if tempo * tempo_estimate_elapsed_time > 2 * Rc:
        if (cur_pos-last_beat_position_2)/100 > 60/Rc * 2:
            # print "tempo changed"
            tempo = tempo_estimate(tempo_estimate_elapsed_time, cur_pos, old_pos,Rc)
            # with bound for tempo
            if tempo / float(Rc) < temp_downbound:
                tempo = Rc * temp_downbound
            elif tempo / float(Rc) > temp_upbound:
                tempo = Rc * temp_upbound
            tempo_estimate_elapsed_time = 0
            estimated_tempo = tempo
            old_pos = cur_pos
            last_beat_position_2 = cur_pos

        if confidence_record_check == 1:
            # confidence_queue.append(confidence[int(cur_time / resolution)])
            confidence_queue.append(confidence[cur_pos])
            confidence_record_check = 0

        # print 'currently at %d' % cur_pos
        # print 'cur_time %f' % cur_time
        print("currenly at"+str(cur_pos))
        print("cur_time " + str(cur_time))
        # print("cur_midipitch" + str(score_midi[cur_pos]))
        # print "end_time %f" % (float(time.clock())-float(start_time))
        # print "origianl_time %f" % real_time

        matched_score.append(score_midi[cur_pos])
        time_axis.append(cur_time - performance_start_time)
        mapped_time.append(cur_pos * resolution)
        tempos.append(tempo)

    new_midi.instruments.append(piano)
    new_midi.write(NEWFILE) 
    # np.savetxt(CONF_FILE, confidence)
    np.savetxt(PITCHFILE, pitchfile)
    # score_plt = [min(-x / 5, 0) for x in score_midi]
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.spines['left'].set_position(('data', 0))
    # ax.spines['bottom'].set_position(('data', 0))
    # ax.spines['right'].set_color('none')
    # ax.spines['top'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    # s, = plt.plot(score_plt, score_axis, color='r')
    # dp, = plt.plot(time_axis, detected_pitches, 'o', color='g', label='detected pitch', markersize=1)
    # mp, = plt.plot(time_axis, mapped_time, 'o', color='b', label='mapped position', markersize=1)
    # plt.xlabel('audio time (seconds)')
    # plt.ylabel('score time (seconds)')
    # plt.grid()
    # plt.show()
    



if __name__ == "__main__":
    # start real time
    # real_time = time.clock()
    score_follow(audio_file=AUDIOFILE, midi_file=MIDIFILE, feature='onset', mask=FILTER)
    # print "Beat list ----------------------"
    # print time_list_for_beat
    # print "confidence_list -------------------------"
    # print confidence_queue
    time_beat_file = open(time_beat_name, 'w')
    confidence_queue_file = open(confidence_name, 'w')
    for item in time_list_for_beat:
        time_beat_file.write(str(item))
        time_beat_file.write("\n")
    for item2 in confidence_queue:
        confidence_queue_file.write(str(item2))
        confidence_queue_file.write("\n")
