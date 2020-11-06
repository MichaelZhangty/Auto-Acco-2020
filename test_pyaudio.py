import pyaudio
import wave
from auto_acco_combine_utilities import *

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(1024)
    frames.append(data)
    data = np.frombuffer(data, dtype=np.int16)
    data = np.true_divide(data, 2 ** 15, dtype=np.float32)
    print("raw",len(data))
    pitch = pitch_detection_aubio(data,2,CHUNK)
    print(pitch)
    # time.sleep(0.05)
    # for i in range(100000):
    #     k = 1+1

    

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()