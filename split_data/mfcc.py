import librosa as lbr
import numpy as np
import glob
import matplotlib.pyplot as plt

filename = '../../raw_data/*'
len_raw_path = len(filename[:-1])
all_categories = map(lambda x: x+'/*', glob.glob(filename))
print all_categories
for folder in all_categories:
    for file_name in glob.glob(folder):
        print file_name 

S_R = 22050
#S_R = 44100

#filename = '../../raw_data/rock/rock.00000.au'
#filename = '../../raw_data/raga1.mp3'
filename = '../../raw_data/kA.mp3'
y, sr = lbr.load(filename, sr=S_R, duration=30.0)
y_harmonic = lbr.effects.harmonic(y)
#y_notes = lbr.hz_to_note(y_harmonic)
#y_harmoic = lbr.note_to_hz(y_notes)
print 'done loading'
end = 600
print y.shape
hop_length = 512
nF = 128
#mfcc = np.log(lbr.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=nF)).T
#mfcc = np.log(lbr.feature.melspectrogram(y=y_harmonic, sr=sr, hop_length=hop_length)).T
#mfcc = lbr.feature.tonnetz(y=y_harmonic, sr=sr).T
mfcc = lbr.feature.chroma_stft(y=y_harmonic, sr=sr).T
#mfcc = mfcc[0:end, :]
print mfcc.shape


filename = '../../raw_data/kAB.mp3'
#filename = '../../raw_data/raga2.mp3'
#filename = '../../raw_data/rock/rock.00008.au'
#filename = '../../raw_data/classical/classical.00008.au'
y, sr = lbr.load(filename, sr=S_R,duration=30.0)
y_harmonic = lbr.effects.harmonic(y)
end = 600
print y.shape
hop_length = 512
nF = 128
#mfcc2 = np.log(lbr.feature.mfcc(y=y_harmonic, sr=sr, hop_length=hop_length, n_mfcc=nF)).T
#mfcc2 = np.log(lbr.feature.melspectrogram(y=y_harmonic, sr=sr, hop_length=hop_length)).T
#mfc2 = lbr.feature.tonnetz(y=y_harmonic, sr=sr).T
mfc2 = lbr.feature.chroma_stft(y=y_harmonic, sr=sr).T
#mfcc2 = mfcc2[0:end, :]
print mfc2.shape

plt.subplot(1, 2, 1)
lbr.display.specshow(mfcc)
plt.subplot(1, 2, 2)
lbr.display.specshow(mfc2)
plt.show()
