import time
import threading
# from auto_accompany_utilities import *
# import fluidsynth
from scipy import stats
import matplotlib.pyplot as plt
from mingus.midi import fluidsynth
from mingus.containers import Note
import mingus.core.notes as notes
from auto_acco_combine_utilities import *


# BPM parameter for each midi
# zhui guang zhe
# BPM = 74
# shuo san jiu san
# BPM = 70
# nanshannan
# BPM = 67


# tempo for the original midi
BPM = 67
Rc = 67
# file names
audio_name = "audio4"
midi_name = "midi4_quick"
# name to save
audio_name_save = "audio4_muti_test"
# audio end time
audio_end_time = 30




# performance end_time
end_time = 1
# audio end time
# wf.getnframes() to get audio time 

audio_file = 'audio/{}.wav'.format(audio_name)
midi_file = 'midi/{}.mid'.format(midi_name)


ACC_FILE = 'midi/{}.mid'.format(midi_name)

BPS = BPM / float(60)  # beat per second
original_begin = time.clock()
global_tempo = 0


# weight is for 0.5beats for 2 beats
weight_judge = True
beat_back = 4
pressed_key = "lol"
timeQueue = []
stop_thread = False
latency_end = -1
resolution = 0.01
sQueue = [1]
score_midi, score_axis, score_onsets, onsets, raw_score_midi = get_time_axis(resolution,midi_file)
score_midi = score_midi_modification(score_midi)
scoreLen = len(score_axis)
fsource = np.zeros(scoreLen)

performance_start_time = 0
wf = wave.open(audio_file, 'rb')
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
n_frames = int(performance_start_time * wf.getframerate())
wf.setpos(n_frames)
length = wf.getnframes()
# confidence should follow the length of audio
confidence = np.zeros(int(math.ceil(wf.getnframes()/441))+1)
time_list_for_beat = [1,2,3,4,5]
confidence_queue = [0.001, 0.001, 0.001, 0.001, 0.001]
score_following_finish = False

# for check score_following
score_following_midi = pretty_midi.PrettyMIDI()
piano_program_following = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
piano_following = pretty_midi.Instrument(program=piano_program_following)
NEWFILE = 'score_following/score_muti_generated_{}.mid'.format(audio_name_save)

# thread for simulating the audio file
def press_key_thread():
    global pressed_key
    global stop_thread
    global latency_end
    global fsource
    global time_list_for_beat
    global score_following_finish
    cnt = 0
    CHUNK = 1024
    time_int = float(CHUNK) / 44100
    cur_time = performance_start_time
    old_time = performance_start_time
    tempo_estimate_elapsed_time = 0
    tempo_estimate_elapsed_time2 = 0

    start_time = time.clock()
    count_cut = 1 
    datas = []
    estimated_tempo = Rc
    confidence_record_check = 0
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
    beat_back = 4

    while not stop_thread:
        while wf.tell() < wf.getnframes():
            while time.clock() - start_time < count_cut * time_int:
                pass

            count_cut += 1
            data = wf.readframes(CHUNK)
            # play the frame
            # stream.write(data)
            if len(data)<2048:
                break
            datas.append(data)
            if len(datas) >= 3:
                c_data = datas[-3:]
                c_data = b''.join(c_data)
                pitch = pitch_detection_aubio(c_data,3)
            else:
                c_data = data
                pitch = pitch_detection_aubio(c_data,1)

            onset_detector = aubio.onset("default", CHUNK, CHUNK, 44100)
            data_onset = np.fromstring(data, dtype=np.int16)
            data_onset = np.true_divide(data_onset, 32768, dtype=np.float32)
 
            if len(data_onset) == 1024: 
                onset_detector(data_onset)
                onset_prob = onset_detector.get_last() / 44100

            tempo_estimate_elapsed_time += time_int
            tempo_estimate_elapsed_time2 += time_int
            cur_time += time_int

            if cur_time > audio_end_time:
                score_following_finish = True
                stop_thread = True
                break

            if pitch != -1:
                pitch = pitch%12

            pitches.append(pitch)
            elapsed_time = cur_time - old_time
            tempo = estimated_tempo


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


            if fsource[cur_pos] > confidence[int(cur_time / resolution)]:
                confidence[int(cur_time / resolution)] = fsource[cur_pos]
            old_time = cur_time

            old_idx = onset_idx
            while onset_idx < len(onsets) and cur_pos >= onsets[onset_idx]:
                onset_idx += 1

            if old_idx < onset_idx:
                old_note = old_midi.instruments[0].notes[onset_idx - 1]
                dur = old_note.end - old_note.start
                new_note = pretty_midi.Note(velocity=old_note.velocity, pitch=old_note.pitch, start=cur_time,
                                        end=cur_time + dur)
                piano_following.notes.append(new_note)
                # print("append ----------------------new note")

            for i in range(len(onsets)):
                if cur_pos < onsets[i]:
                    break
                old_idx = i - 1

            # record the beat time
            if tempo * tempo_estimate_elapsed_time2 > 1 * Rc:
                time_list_for_beat.append(time_list_for_beat[-1] + tempo_estimate_elapsed_time2)  # list can be improved
                tempo_estimate_elapsed_time2 = 0
                confidence_queue.append(confidence[int(cur_time / resolution)])

            if tempo * tempo_estimate_elapsed_time > 2 * Rc:
                tempo = tempo_estimate(tempo_estimate_elapsed_time, cur_pos, old_pos,Rc,resolution)
                if tempo / float(Rc) < temp_downbound:
                    tempo = Rc * temp_downbound
                elif tempo / float(Rc) > temp_upbound:
                    tempo = Rc * temp_upbound
                tempo_estimate_elapsed_time = 0 
                estimated_tempo = tempo
                old_pos = cur_pos

                if latency_end == -1:
                    l = 0.1
                else:
                    l = max(0, latency_end - time.clock())
                b0, t0, s0 = compute_tempo_ratio_weighted(b0, t0, s0, l,time_list_for_beat,beat_back,confidence_queue)
                sQueue.append(s0)
                # print(sQueue)
        
            # print("cur_time " + str(cur_time))
            # print("start_time"+ str(start_time))
            # print("real_time_SSSSSS------" + str(time.clock()-start_time))
            # print("cur_midipitch" + str(score_midi[cur_pos]))
            # print "end_time %f" % (float(time.clock())-float(start_time))
        score_following_midi.instruments.append(piano_following)
        score_following_midi.write(NEWFILE) 
        print("start_time"+ str(start_time))

    score_following_finish = True
    stop_thread = True
    print("finish score following")


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
        # score_midi, axis, score_onsets, onsets, raw_score_midi
        self.score_midi, self.axis, self.score_onsets, self.onsets,self.raw_score_midi = get_time_axis(self.resolution, self.ACC_FILE)
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
        global time_list_for_beat
        global latency_end
        global score_following_finish
        begin = time.clock()
        print("begin time------------------------------" + str(begin))
        total_delay = 0

        for i in range(start, len(self.notes)):
            note = self.notes[i]
            if i == 0:
                if note.start - time.clock() + begin + total_delay - 0.1 > 0:
                    time.sleep(note.start - time.clock() + begin + total_delay - 0.1)
            cur_time = time.clock() - begin - total_delay
            wait_delta = note.start - cur_time
            # if cur_time > end_time:
            #     break
            if score_following_finish:
                break
            # print(sQueue)
            tempo_ratio = float(self.BPS) / sQueue[-1]
            # print(sQueue)
            # print("cur_time_acco--------------------------" + str(cur_time))
            # print ("Tempo_ratio == "+str(tempo_ratio))
            total_delay += wait_delta * (tempo_ratio-1)
            wait_delta = wait_delta * tempo_ratio


            target_start_time = time.clock() + wait_delta
            latency_end = target_start_time
            # print("real_AAAA------------" + str(time.clock()))
            # print("target_wake_up----"+ str(target_start_time-begin))
            try:
                if target_start_time > time.clock():
                    time.sleep(target_start_time-time.clock())
            except:
                break
            # print("wake_up----" + str(time.clock()-begin))

            self.playTimes.append(time.clock() - original_begin)
            self.noteTimes.append(note.start)
            # play the note
            # n = Note(notes.int_to_note(note.pitch%12),note.pitch//12,300)
            # fluidsynth.play_Note(n)

            tempo_ratio = float(self.BPS) / sQueue[-1]

            cur_time = time.clock() - begin - total_delay

            new_note = pretty_midi.Note(velocity=note.velocity, pitch=note.pitch, start=cur_time, end=cur_time + note.end - note.start)
            # print(str(cur_time)+"-------------------------")
            # print(str(cur_time + note.end - note.start))
            piano.notes.append(new_note)

            delta_time = note.end - (time.clock() - begin - total_delay)
            # print "---------------------------correct %f" %time.clock()
            total_delay += delta_time * (tempo_ratio - 1)
            delta_time = delta_time * tempo_ratio

            target_time = time.clock() + delta_time
            latency_end = target_time

            try:
                if target_time > time.clock():
                    time.sleep(target_time-time.clock())
            except:
                # print("break for time sleep")
                break

            # self.fs.noteoff(0, note.pitch)

            old_target_start_time = target_start_time + 0

        stop_thread = True


        tap_time = [t-5 for t in time_list_for_beat[4:]]
        # print("ready to plot")
        # tap_beat = [(t-4) / float(self.BPS) for t in range(len(tap_time))]
        tap_beat = [(t)/ float(self.BPS) for t in range(len(tap_time))]


        new_midi.instruments.append(piano)
        new_midi.write("auto_accompany/auto_accompany{}.mid".format(audio_name_save))



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
    fs = 1
    fluidsynth.init("soundfont.SF2")
    # fs = fluidsynth.Synth()
    # sfid = fs.sfload("soundfont.sf2")
    # fs.start("coreaudio")
    # fs.program_select(0, sfid, 0, 0)
    try:
        player = Player(ACC_FILE, original_begin, BPM, fs)
        player.follow(0)
        print("finish follow")
    except KeyboardInterrupt:
        # stop_thread = True
        # pk_thread.killed = True
        # _thread.exit()
        # pk_thread.terminate()
        pk_thread.join()
        # fs.delete()
    finally:
        # stop_thread = True
        # pk_thread.killed = True
        # _thread.exit()
        # pk_thread.terminate()
        pk_thread.join()
        # fs.delete()
