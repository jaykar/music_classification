import librosa as lbr
import numpy as np
import glob
import matplotlib.pyplot as plt

#params for sampling rate of 22050Hz for 30 seconds
S_R = 22050
TIME = 1292
FREQ = 128
hop_length = 512

def get_mcc(filename):
    y, sr = lbr.load(filename, sr=S_R, duration=30.0)
    mcc = np.log(lbr.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)).T
    return mcc

filename = '../../raw_data/*'
len_raw_path = len(filename[:-1])
all_categories = map(lambda x: x+'/*', glob.glob(filename))
all_categories = [filename[:-1]+'classical/*', filename[:-1]+'rock/*', filename[:-1] + 'jazz/*']

n_songs = 100
n_points = len(all_categories)* n_songs

X = np.zeros((n_points, TIME, FREQ))
Y = np.zeros((n_points,1)) 
i = 0
label = 0
for folder in all_categories:
    for file_name in glob.glob(folder):
        print file_name 
        X[i,:,:] = get_mcc(file_name)
        Y[i, 0] = label
        i += 1
    if i %100 == 0:
        label += 1
        print(label)


np.save('X.npy', X)
np.save('Y.npy', Y)

#filename = '../../raw_data/rock/rock.00000.au'
#filename = '../../raw_data/raga1.mp3'
#filename = '../../raw_data/kA.mp3'
#y, sr = lbr.load(filename, sr=S_R, duration=30.0)
#y_harmonic = lbr.effects.harmonic(y)
#y_notes = lbr.hz_to_note(y_harmonic)
#y_harmoic = lbr.note_to_hz(y_notes)
#print 'done loading'
#end = 600
#print y.shape
#nF = 128
#mfcc = np.log(lbr.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=nF)).T
#mfcc = np.log(lbr.feature.melspectrogram(y=y_harmonic, sr=sr, hop_length=hop_length)).T
#print mfcc
#mfcc = lbr.feature.tonnetz(y=y_harmonic, sr=sr).T
#mfcc = lbr.feature.chroma_stft(y=y_harmonic, sr=sr).T
#mfcc = mfcc[0:end, :]
#print mfcc.shape


#filename = '../../raw_data/kAB.mp3'
#filename = '../../raw_data/raga2.mp3'
#filename = '../../raw_data/rock/rock.00008.au'
#filename = '../../raw_data/classical/classical.00008.au'
#y, sr = lbr.load(filename, sr=S_R,duration=30.0)
#y_harmonic = lbr.effects.harmonic(y)
#end = 600
#print y.shape
#hop_length = 512
#nF = 128
#mfcc2 = np.log(lbr.feature.mfcc(y=y_harmonic, sr=sr, hop_length=hop_length, n_mfcc=nF)).T
#mfcc2 = np.log(lbr.feature.melspectrogram(y=y_harmonic, sr=sr, hop_length=hop_length)).T
#mfc2 = lbr.feature.tonnetz(y=y_harmonic, sr=sr).T
#mfc2 = lbr.feature.chroma_stft(y=y_harmonic, sr=sr).T
#mfcc2 = mfcc2[0:end, :]
#print mfc2.shape

#plt.subplot(1, 2, 1)
#lbr.display.specshow(mfcc)
#plt.subplot(1, 2, 2)
#lbr.display.specshow(mfc2)
#plt.show()
