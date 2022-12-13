# import all required libraries

import sounddevice as sound
from scipy.io.wavfile import write
import wavio as wv


class voice_record():
    def record(self):
        freq = 44100
        duration=2
        print("speak")
        recording = sound.rec(int(duration*freq),
                          samplerate=freq,channels=2)
        sound.wait()

        write('recording.wav',freq,recording)
        return recording
# a= voice_record()
# a.record()
        
