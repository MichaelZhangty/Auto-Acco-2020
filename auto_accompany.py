import time
import threading
from auto_accompany_utilities import *
import fluidsynth
from scipy import stats
import matplotlib.pyplot as plt


# BPM parameter for each midi
# zhui guang zhe
# BPM = 74
# shuo san jiu san
# BPM = 70
# nanshannan
# BPM = 67

BPM = 70


# name of the audio file
audio_name = "shuosanjiusan_gus_6_9"
midi_name = "midi3"
end_time = 60


time_beat_file = 'time_beat/time_beat_{}.txt'.format(audio_name)
confidence_file = 'confidence/confidence_queue_{}.txt'.format(audio_name)
ACC_FILE = 'midi/{}.mid'.format(midi_name)


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






BPS = BPM / float(60)  # beat per second
original_begin = time.clock()
global_tempo = 0

# weight is for 0.5beats for 2 beats
weight_judge = True
beat_back = 4
pressed_key = "lol"
timeQueue = []
stop_thread = False
sQueue = []
latency_end = -1

# for simulation
simulation_times = [1, 2, 3, 4, 5]
for i in time_list_for_beat:
    simulation_times.append(i + 5)


# thread for simulating the audio file
def press_key_thread():
    global pressed_key
    global stop_thread
    global latency_end
    cnt = 0
    while not stop_thread:
        if cnt < len(simulation_times) and (
                 abs(time.clock() - original_begin - simulation_times[cnt])) <= 0.003 or time.clock() - original_begin >= \
                 simulation_times[cnt]:
            timeQueue.append(time.clock() - original_begin)
            cnt = cnt + 1 
            if len(timeQueue) == 5:
                if weight_judge:
                    b0 = 1
                    t0 = timeQueue[-1]
                    x = timeQueue[-beat_back:]
                    y = range(1, beat_back + 1)
                    confidence_block = confidence_queue[len(timeQueue) - beat_back:len(timeQueue) - 0]
                    sum_confidence = sum(confidence_block)
                    confidence_count = []
                    for index in range(len(confidence_block)):
                        confidence_count.append(round(confidence_block[index] / sum_confidence, 1))
                    for index in range(len(confidence_count)):
                        print confidence_count[index]
                        times = int(confidence_count[index] * 10)
                        for i in range(times - 1):
                            x.append(x[index])
                            y.append(y[index])
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
                    b0, t0, s0 = compute_tempo_ratio_weighted(b0, t0, s0, l,timeQueue,beat_back,confidence_queue)
                else:
                    b0, t0, s0 = compute_tempo_ratio(b0, t0, s0, l,timeQueue,beat_back)
                sQueue.append(s0)


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
        global timeQueue
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


        tap_time = [t for t in timeQueue]
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
