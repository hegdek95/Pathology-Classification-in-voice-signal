# %%
from scipy import signal
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

# %%
overlap = 1024
frame_length = 2048

# %%
from scipy.io import wavfile

def readAudio(audio):
    fs, amp = wavfile.read(audio)
    dt = 1/fs
    n = len(amp)
    t = dt*n

    if t > 1.0:
        amp = amp[int((t/2 - 0.5)/dt):int((t/2 + 0.5)/dt)]
        n = len(amp)
        t = dt*n
    
    return(amp, fs, n, t)

# %%
amp, fs, n ,t = readAudio('1205-a_h.wav')

fig = plt.figure(figsize=(10,4))

plt.plot(amp)
plt.ylabel('amplitude')
plt.xlabel('sample')
plt.title('signal')

# %%
S = librosa.feature.melspectrogram(y=amp*1.0, sr=fs, n_fft=frame_length, hop_length=overlap, power=1.0)
fig = plt.figure(figsize=(10,4))

librosa.display.specshow(librosa.power_to_db(S,ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()

# %%
import cv2

print('original shape: ', librosa.power_to_db(S,ref=np.max).shape)
img = cv2.resize(librosa.power_to_db(S,ref=np.max),(64,64))
librosa.display.specshow(img, y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()

# %%
print(librosa.power_to_db(S,ref=np.max).shape)
import numpy as np

dS = np.gradient(librosa.power_to_db(S,ref=np.max), axis=0)
dS.shape
ddS = np.gradient(dS, axis=0)

# %%
img = cv2.resize(dS,(64,64))
librosa.display.specshow(img, y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

# %%
amp, fs, n, t = readAudio('./data/Laryngozele/1205-a_h.wav')

fig = plt.figure(figsize=(10,4))

plt.plot(amp)
plt.ylabel('amplitude')
plt.xlabel('sample')
plt.title('signal')

# %%
S = librosa.feature.melspectrogram(y=amp*1.0, sr=fs, n_fft=frame_length, hop_length=overlap, power=1.0, fmin=500.0)

fig = plt.figure(figsize=(10,4))
librosa.display.specshow(librosa.power_to_db(S,ref=np.max), y_axis='mel', fmax=8000, fmin=500.0, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

# %%
img = cv2.resize(librosa.power_to_db(S,ref=np.max),(64,64))
librosa.display.specshow(img, y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

# %%
print(librosa.power_to_db(S,ref=np.max).shape)
import numpy as np

dS = np.gradient(librosa.power_to_db(S,ref=np.max), axis=0)
dS.shape
ddS = np.gradient(dS, axis=0)

# %%
img = cv2.resize(dS,(64,64))
librosa.display.specshow(img, y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

# %%

