import numpy as np
import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import sklearn
import IPython

joel,sr = librosa.load("/mnt/data/datasets/medleyDB/V1/LizNelson_Rainfall/LizNelson_Rainfall_MIX.wav")

joel = joel[10*sr:(40)*sr]

stft = librosa.stft(joel,n_fft = int(0.1*sr),hop_length=256)

mag,ph = librosa.magphase(stft)

psd = np.abs(mag)**2

time_bins = psd.shape[1]
freq_bins = psd.shape[0]



Ust = 5
F = psd.shape[0]
Fmin = librosa.note_to_hz('C2')
Fmax = librosa.note_to_hz('C7')
#Fmax = 20000

U = int(np.log2(Fmax/Fmin) * 12*Ust +1)

freq=librosa.fft_frequencies(sr,0.1*sr)

def uf(u,Fmin,Ust):
    return Fmin * np.power(2,(u)/(12*Ust))


u = np.arange(U)
uf0 = uf(u,Fmin,Ust)

var = 100 #hertz square

Wf0 = np.load('WF0.npz')
Wf0 = Wf0['arr_0']

time = np.arange(time_bins)

K = 10
P = 30

W_G = np.load('Wgamma.npz')['arr_0']

H_G = np.random.rand(P,K)
H_phi = np.random.rand(K,time_bins)
WM = np.random.rand(F,U)
HM = np.random.rand(U,time_bins)
Hf0 = np.random.rand(U,time_bins)

np.savez('stuff_liz.npz',Wf0,W_G,H_G,H_phi, WM, Hf0, HM, psd)