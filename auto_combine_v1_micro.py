import time
import threading
from scipy import stats
import matplotlib.pyplot as plt
from auto_acco_combine_utilities import *
import fluidsynth


# BPM parameter for each midi
# zhui guang zhe
# BPM = 74
# shuo san jiu san
# BPM = 70
# nanshannan
# BPM = 67


# tempo for the original midi
BPM = 70
Rc = 70
# file names
audio_name = '''audio3'''
midi_name = "midi3"
# name to save
audio_name_save = "audio3"
# audio end time
audio_end_time = 30


play = True
audio_file = 'audio/{}.wav'.format(audio_name)
midi_file = 'midi/{}.mid'.format(midi_name)
ACC_FILE = 'midi/{}.mid'.format(midi_name)
BPS = BPM / float(60)  # beat per second
# weight is for 0.5beats for 2 beats
stop_thread = False
resolution = 0.01 #0.01
score_midi, score_axis, score_onsets, onsets, raw_score_midi,axis_loudness = get_time_axis(resolution,midi_file)
scoreLen = len(score_axis)
fsource = np.zeros(scoreLen)
# wf = wave.open(audio_file, 'rb')
p = pyaudio.PyAudio()
# stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#                     channels=wf.getnchannels(),
#                     rate=wf.getframerate(),
#                     output=True,frames_per_buffer=1024) #1024
# for micro input 
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=2048) # 1024
performance_start_time = 0
# n_frames = int(performance_start_time * wf.getframerate())
# wf.setpos(n_frames)
# length = wf.getnframes()
confidence = np.zeros(scoreLen)# confidence should follow the length of audio
time_list_for_beat = [0,1,2,3,4]
beat_list = [0,1,2,3,4]
confidence_queue = [0.001, 0.001, 0.001, 0.001, 0.001]
score_following_finish = False
score_following_midi = pretty_midi.PrettyMIDI()
piano_program_following = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
piano_following = pretty_midi.Instrument(program=piano_program_following)
NEWFILE = 'score_following/score_muti_generated_{}.mid'.format(audio_name_save)
# thread for simulating the audio file
def press_key_thread():
    global stop_thread
    global latency_end
    global fsource
    global time_list_for_beat
    global score_following_finish
    global sQueue
    global cur_pos
    global start_time
    global begin
    latency_end = -1
    CHUNK = 1024 #1024
    time_int = float(CHUNK) / 44100
    cur_time = performance_start_time
    old_time = performance_start_time
    tempo_estimate_elapsed_time = 0
    tempo_estimate_elapsed_time2 = 0

    count_cut = 1 
    datas = []
    estimated_tempo = Rc
    temp_downbound = 0.7 #0.7
    temp_upbound = 1.3 #1.3
    no_move_flag = False
    fsource[0] = 1
    cur_pos = 0
    old_pos = 0
    alpha = 10
    pitches = []
    onset_idx = 0
    b0 = 4
    t0 = time_list_for_beat[-1]
    s0 = float(1) / (time_list_for_beat[-1] - time_list_for_beat[-2])
    sQueue = [s0]
    beat_back = 4
    # to record b0 t0
    list_b0 = [b0]
    list_t0 = [t0]

    # while not stop_thread:
    last_beat_position_1 = 0
    last_beat_position_2 = 0
    # print("start_time"+str(time.clock()))
    time.sleep(0.23)
    start_time = time.clock()
    print("start_time---------------------------"+str(time.clock()))
    print("Let's goooooooooooo!!!!!!")
    while cur_time <= audio_end_time:
            # data = wf.readframes(CHUNK)
            bytes_read = stream.read(1024)
            data = np.frombuffer(bytes_read, dtype=np.float32)
            # print(str(stream.get_input_latency())+"input latency")
            # print(str(stream.get_output_latency())+"output latency")
            # stream.write(data)
            # while time.clock() - start_time < count_cut * time_int:
            #     # print("stuck--------------------------------")
            #     pass
            # if time.clock() - start_time < count_cut * time_int:
            #     time.sleep(count_cut * time_int-time.clock()+start_time)

            count_cut += 1
            # data = wf.readframes(CHUNK)
            # play the frame
            # if play:
            #     stream.write(data)
            # if len(data) < CHUNK*2:
            #     break
            datas.append(data)
            # if len(datas) >= 3:
            #     c_data = datas[-3:]
            #     c_data = b''.join(c_data)
            #     pitch = pitch_detection_aubio(c_data,3,CHUNK)
            # else:
            #     c_data = data
            #     pitch = pitch_detection_aubio(c_data,1,CHUNK)
            pitch = pitch_detection_aubio(data,1,CHUNK)
            data_onset = data

            onset_detector = aubio.onset("default", CHUNK, CHUNK, 44100)
            # data_onset = np.fromstring(data, dtype=np.int16)
            # data_onset = np.true_divide(data_onset, 32768, dtype=np.float32)
 
            if len(data_onset) == CHUNK: #1024
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
            # print("curren_time " + str(cur_time) + "pitch is " + str(pitch) + "jump_count  " + str(jump_count))

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
            gate_mask = create_gate_mask(cur_pos, scoreLen,tempo,Rc)
            fsource = fsource * gate_mask
            fsource = fsource / sum(fsource)
            cur_pos = np.argmax(fsource)
            # print("cur_pos is " + str(cur_pos))

            # for suddenly slient while singing in the middle       
            if pitch == -1 and score_midi[cur_pos]!= -1:
                # # try new method to help the end
                # check = True
                # for index in range(100):
                #     if score_midi[cur_pos+index] == -1:
                #         check = False
                #         break
                # if check:
                #     no_move_flag = True
                # else:
                no_move_flag = False
            # for stuck after long slience to wait for some sound
            if pitch == -1 and score_midi[cur_pos+1]!= -1:
                no_move_flag = True
            else:
                no_move_flag = False
            # print("no_move_flag")



            # no_move_flag = loudness == 1 and axis_loudness[cur_pos] + axis_loudness[cur_pos + 1] == 2




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
                # start_sign = True
                # if jump_count >= 20:
                #     print("jump_count"+str(jump_count))
                #     print("current_pitch" + str(pitch))
                #     jump_sign = True
                    # print("jump!!!!!!!"+str(cur_time))
                    # print("cur_time " + str(cur_time))
            # else:
            #     # jump_count += 1
            #     count_delay += 1
            #     if count_delay >= 3 and jump_sign:
            #         print("miss jump sign" + str(cur_time))
            #         jump_sign = False
            #         count_delay = 0

            for i in range(len(onsets)):
                if cur_pos < onsets[i]:
                    break
                old_idx = i - 1

            # record the beat time
            if (cur_pos-last_beat_position_1)/100 > 45/Rc: #60/Rc #45 good
                time_list_for_beat.append(time_list_for_beat[-1] + tempo_estimate_elapsed_time2)  # list can be improved
                tempo_estimate_elapsed_time2 = 0
                confidence_queue.append(confidence[int(cur_time / resolution)])
                beat_list.append(beat_list[-1]+((cur_pos-last_beat_position_1)/100)/(60/Rc))
                last_beat_position_1 = cur_pos

                # change tempo for accompany
                if latency_end == -1:
                    l = 0.1
                else:
                    l = max(0, latency_end - time.clock())
                # print("latency=========== " + str(l))
                b0, t0, s0 = compute_tempo_ratio_weighted(b0, t0, s0, l,time_list_for_beat,beat_back,confidence_queue,beat_list)
                # b0, t0, s0 = compute_tempo_ratio(b0,t0,s0, l,time_list_for_beat)
                list_b0.append(b0)
                list_t0.append(t0)
                sQueue.append(s0)
            
                

            # if tempo * tempo_estimate_elapsed_time > 2 * 60: #rc
            if (cur_pos-last_beat_position_2)/100 > 60/Rc * 2:
                tempo = tempo_estimate(tempo_estimate_elapsed_time, cur_pos, old_pos,Rc,resolution)
                if tempo / float(Rc) < temp_downbound:
                    tempo = Rc * temp_downbound
                elif tempo / float(Rc) > temp_upbound:
                    tempo = Rc * temp_upbound
                tempo_estimate_elapsed_time = 0 
                estimated_tempo = tempo
                old_pos = cur_pos
                last_beat_position_2 = cur_pos


                # print(sQueue)
                # print("score_tempo " + str(tempo))
                # print("acco_tempo " + str(s0*60))
            # print("cur_time_score_following_midi"+str(cur_pos*100))
            # print("cur_time " + str(cur_time))
            # print("start_time"+ str(start_time))
            # print("real_time_SSSSSS------" + str(time.clock()-start_time))
            # print("cur_midipitch" + str(score_midi[cur_pos]))
            # print "end_time %f" % (float(time.clock())-float(start_time))

    t0_file = open("time_beat/t0file_4beat",'w')
    b0_file = open("time_beat/b0file_4beat",'w')
    time_list_file = open("time_beat/timefile",'w')
    beat_file = open("time_beat/beatfile","w")
    for item in list_b0:
        b0_file.write(str(item)+"\n")
    for item2 in list_t0:
        t0_file.write(str(item2)+"\n")
    for item3 in time_list_for_beat:
        time_list_file.write(str(item3)+"\n")
    for item4 in beat_list:
        beat_file.write(str(item4)+"\n")
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
    def __init__(self, ACC_FILE, BPM, fs):
        self.ACC_FILE = ACC_FILE
        self.start_time = 0 #start_time
        self.midi_data = pretty_midi.PrettyMIDI(ACC_FILE)
        self.notes = sorted(self.midi_data.instruments[0].notes, key=lambda x: x.start, reverse=False)
        self.resolution = 0.1
        # self.end_time = end_time
        # score_midi, axis, score_onsets, onsets, raw_score_midi
        self.score_midi, self.axis, self.score_onsets, self.onsets,self.raw_score_midi = get_time_axis_auto_acco(self.resolution, self.ACC_FILE)
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
        global cur_pos
        global start_time
        global begin
        # global start_sign
        begin = time.clock()
        print("begin time------------------------------" + str(begin))
        total_delay = 0
        cnt_accompany = 0
        last_note_pitch = -1
        note_to_append_time = 0
        cnt_note_amount = 0

        # for i in self.raw_score_midi:
        index = 0
        # self.start_time = start_time
        schedule_time = begin # begin
        # pitch_time_list = []
        # pitch_play_list = []
        # pitch_play_index = 0
        # fs.noteon(0,40,100)
        while index < len(self.raw_score_midi):       
                note_pitch = self.raw_score_midi[index]
                if score_following_finish:
                    break
                s = sQueue[-1]
                if s == 0:
                    s = s_old
                else:
                    s_old = s
                tempo_ratio = float(self.BPS)/s
                schedule_time += self.resolution * tempo_ratio
                target_start_time = schedule_time
                latency_end = schedule_time
                print("tempo_ratio ---------------------" + str(tempo_ratio))


                # smooth version
                # note_cur_time = time.clock() - begin
                note_cur_time = time.clock() - begin
                print("cur_time_accompany_midi"+str(index*self.resolution))
                print("cur_time_score_following_midi"+str(cur_pos/100))
                # print("real_time_AAAAAAAA"+str(time.clock()-begin))
                print("pitch----------------" + str(note_pitch))


                if note_pitch > -1:
                    if last_note_pitch != note_pitch and last_note_pitch > -1:
                        if play:
                            fs.noteoff(0,last_note_pitch)
                        new_note = pretty_midi.Note(velocity=100, pitch=last_note_pitch, start=note_to_append_time, end=note_cur_time)
                        piano.notes.append(new_note)
                        self.playTimes.append(note_to_append_time)
                        self.noteTimes.append((cnt_accompany-cnt_note_amount)*self.resolution)
                        # cnt_note_amount = 1
                        if play:
                            fs.noteon(0,note_pitch,100)
                        note_to_append_time = note_cur_time
                    elif last_note_pitch == note_pitch:
                        cnt_note_amount += 1
                    elif last_note_pitch != note_pitch and last_note_pitch == -1:
                        if play:
                            fs.noteon(0,note_pitch,100)
                        note_to_append_time = note_cur_time
                        cnt_note_amount = 1
                else:
                    if last_note_pitch > -1:
                        if play:
                            fs.noteoff(0,last_note_pitch)
                        new_note = pretty_midi.Note(velocity=100, pitch=last_note_pitch, start=note_to_append_time, end=note_cur_time)
                        piano.notes.append(new_note)
                        self.playTimes.append(note_to_append_time)
                        self.noteTimes.append((cnt_accompany-cnt_note_amount)*self.resolution)
                        cnt_note_amount = 0

                # fs.noteon(1,note_pitch,100)

                try:
                    if target_start_time > time.clock():
                        time.sleep(target_start_time-time.clock())
                        # print("wake up " + str(time.clock()))
                except:
                    break

                # fs.noteoff(1,note_pitch)
                
                last_note_pitch = note_pitch
                cnt_accompany += 1
                index += 1
                # if start_sign or note_pitch == -1:
                #     index += 1




        stop_thread = True


        tap_time = [t-5 for t in time_list_for_beat[4:]]
        # print("ready to plot")
        # tap_beat = [(t-4) / float(self.BPS) for t in range(len(tap_time))]
        tap_beat = [t-5 for t in beat_list[4:]]


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
    # fs = fluidsynth.Synth()
    # fs.start("coreaudio")

    # sfid = fs.sfload("soundfont.sf2")
    # fs.program_select(0, sfid, 0, 0)
    # fs = 1
    # fluidsynth.init("soundfont.SF2")
    fs = fluidsynth.Synth()
    fs.start("coreaudio")
    sfid = fs.sfload("soundfont.sf2")
    fs.program_select(0, sfid, 0, 0)
    try:
        player = Player(ACC_FILE, BPM, fs)
        player.follow(0)
        print("finish follow")
    except KeyboardInterrupt:
        stop_thread = True
        pk_thread.join()
        # fs.delete()
    finally:
        stop_thread = True
        # pk_thread.killed = True
        # _thread.exit()
        # pk_thread.terminate()
        pk_thread.join()
        # fs.delete()

        # fs.delete()
