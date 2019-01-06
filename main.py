import sounddevice as sd
import pickle
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read as wavread

fn = "/Users/derek/Downloads/FDS test 1.wav"
a = wavread(fn)
myrecording = np.array(a[1]/32000,dtype=float)
fs = a[0]

duration = len(myrecording[:,0])

fn = "/Users/derek/Downloads/cAR 9mm FDS 2.wav"
a = wavread(fn)
mytemplate = np.array(a[1]/32000,dtype=float)

# mytemplate = myrecording[100000:140000,0]


# sd.default.samplerate = fs

# duration = 5  # seconds
# fs = 44100
# myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
# sd.wait()
#
# with open("test.pkl", "wb") as f:
#     print("Pickling")
#     pickle.dump(myrecording, f)

# with open("test.pkl", "rb") as f:
#     print("Unpickling")
#     myrecording = pickle.load(f)

# sd.play(myrecording, fs)
# sd.wait()


# print(myrecording)

# plt.plot()
fig,ax = plt.subplots(3, 1)
ax[0].plot(myrecording[:,0])
ax[1].plot(myrecording[:,1])

plt.xlabel('time (s)')
plt.ylabel('audio')
plt.title('What I just said')

sd.play(mytemplate[:int(0.360*fs),0], fs)
sd.wait()

n = (abs(myrecording) > 1.0).sum()

print( "Seconds loud: {}".format(n/fs) )

# Find events with specific duration and volume

crosscorr = signal.fftconvolve(myrecording[:,0],
                               mytemplate[:int(0.360*fs),0], mode='full')

n = (abs(crosscorr) > 1.0).sum()
print( "Seconds corr: {}".format(n/fs) )

ax[2].plot(crosscorr)
plt.ylim(-1000, 1000)


plt.grid(True)

plt.show()

