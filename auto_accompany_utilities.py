
import math
import numpy as np
import pretty_midi
import statsmodels.api as sm

def compute_tempo_ratio(b0, t0, s0, l,timeQueue,beat_back):
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
    return bn, tn, sn



def compute_tempo_ratio_weighted(b0, t0, s0, l,timeQueue,beat_back,confidence_queue):
    t1 = timeQueue[-1]
    b = b0 + (t1 + l - t0) * s0
    tn = t1 + l
    print "latency l %f---------------------------------"%l
    bn = b
    te = timeQueue[-2]
    be = len(timeQueue) - 5
    x = timeQueue[-beat_back:]
    y = range(len(timeQueue) - beat_back - 3, len(timeQueue) - 3)
    confidence_block = confidence_queue[len(timeQueue) - beat_back:len(timeQueue) - 0]
    x = sm.add_constant(x)
    if y[0] == 0:
        wls_model = sm.WLS(y, x)
        results = wls_model.fit()
        se = results.params[1]
    else:
        wls_model = sm.WLS(y, x, weights=confidence_block)
        results = wls_model.fit()
        se = results.params[1]
    print "------------------------------sk-----------------------"
    print se
    sn = (float(4) / (te * se - tn * se - be + bn + 4)) * se
    return bn, tn, sn

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