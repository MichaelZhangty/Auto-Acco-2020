import math
import numpy as np
import pretty_midi
import analyse
from madmom.features.onsets import CNNOnsetProcessor
from scipy.integrate import quad
import aubio
import matplotlib.pyplot as plt

def normpdf(x, mean, sd=1):
    var = float(sd) ** 2
    pi = 3.1415926
    denom = (2 * pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom

# get time axis from midi file
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
                # regulate to 12 pitch
                score_midi[j] = note.pitch - note.pitch / 12 * 12
    # plot to check
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


def compute_f_V_given_I(pitch, pitches, scoreLen, score_midi, onset_prob, score_onsets, alpha, feature, w1, w2, w3,
std=1,WINSIZE = 1,WEIGHT=[0.5]):
    f_V_given_I = np.zeros(scoreLen)
    sims = np.zeros(scoreLen)
    # print("before pitch is " + str(pitch))
    # print(pitches[-1-WINSIZE:-1])
    # print(WEIGHT)
    if pitch == 0:
        pitch = pitch
    elif len(pitches) > WINSIZE:
        pitch = pitch - np.dot(pitches[-1 - WINSIZE:-1], WEIGHT)
    # elif len(pitches) > 1:
    #     pitch = pitch - float(sum(pitches[:-1])) / (len(pitches) - 1)
    # else:
    #     pitch = 0

    # print("after pitch is "+str(pitch))

    for i in range(scoreLen):

        # if score_midi[i] == 0:
        #     score_pitch = 0
        if pitch == 0:
            score_pitch = score_midi[i]
        elif i >= WINSIZE:
            score_pitch = score_midi[i] - np.dot(score_midi[i - WINSIZE:i], WEIGHT)
        # elif i > 0:
        #     score_pitch = score_midi[i] - float(sum(score_midi[:i])) / i
        else:
            score_pitch = score_midi[i]

        score_onset = score_onsets[i]
        if feature == 'onset':
            # for fix no sound bug
            if pitch == 0:
                if score_pitch == 0:
                    f_V_given_I[i] = 0.1
                else:
                    f_V_given_I[i] = 0.00000000001
            elif score_pitch == 0:
                f_V_given_I[i] = 0.00000000001
            elif abs(pitch-score_pitch)<6:
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

    # plt.plot(f_V_given_I[:50])
    # plt.show()

    return f_V_given_I, sims


def compute_f_I_J_given_D(score_axis, estimated_tempo, elapsed_time, beta,alpha,Rc,no_move_flag):
    if no_move_flag:
        print("no move")
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

def pitch_detection(data):
    samps = np.fromstring(data, dtype=np.int16)
    pitch = analyse.musical_detect_pitch(samps)
    # to do: too high situation
    if analyse.loudness(samps) > -25 and pitch != None:
        return pitch
    else:
        return -1
CHUNK = 1024
# CHUNK = 1412
pitch_detector = aubio.pitch('yinfft', CHUNK, CHUNK, 44100)
pitch_detector.set_unit('midi')
pitch_detector.set_tolerance(0.75)

def pitch_detection_aubio(data):
    samps = np.fromstring(data, dtype=np.int16)
    samps = np.true_divide(samps, 32768, dtype=np.float32)
    pitch = pitch_detector(samps)[0]
    if pitch > 84 or pitch < 40:
        return -1
    else:
        return pitch

def tempo_estimate(elapsed_time, cur_pos, old_pos,Rc,resolution=0.01):
    # print 'cur pos %d old pos %d'%(cur_pos,old_pos)
    # print 'elapsed_time %f'%elapsed_time
    return float(cur_pos - old_pos) * Rc * resolution / elapsed_time

window_size = 1000
def compute_f_I_given_D(fsource, f_I_J_given_D, cur_pos, scoreLen):
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

# need to refine (Fake real time)
def plot_onsets_prob(onset_audio_file, scoreLen):
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
    # plt.show()source
    return mask

    