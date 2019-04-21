import numpy as np
import librosa
import os


def generate(path,dest = './outs/'):

    np.random.seed(0)

    joel,sr = librosa.load(path)
    #print(path)
    #"/mnt/data/datasets/medleyDB/V1/MusicDelta_LatinJazz/MusicDelta_LatinJazz_MIX.wav")

    #joel = joel[10*sr:(40)*sr]

    stft = librosa.stft(joel,n_fft = int(0.1*sr),hop_length=220)

    mag,ph = librosa.magphase(stft)

    psd = np.abs(mag)**2

    # hat = np.random.rand(psd.shape[0],psd.shape[1])

    # print(-np.log(psd/hat))
    # print(psd/hat)

    eps = 10**-30
    for i in range(psd.shape[0]):
        for j in range(psd.shape[1]):
            if psd[i,j] == 0.000:
                #print(i,j)
                psd[i,j] = eps

    #print(psd)
    # print(-np.log(psd/hat))
    # print(psd/hat)

    time_bins = psd.shape[1]
    # F = freq_bins = psd.shape[0]



    # Ust = 5
    # F = 1103 #psd.shape[0], constant for constant sr
    # Fmin = librosa.note_to_hz('C2')
    # Fmax = librosa.note_to_hz('C7')
    #
    # U = int(np.log2(Fmax/Fmin) * 12*Ust +1)
    # #
    # # freq=librosa.fft_frequencies(sr,0.1*sr)
    # #
    # # def uf(u,Fmin,Ust):
    # #     return Fmin * np.power(2,(u)/(12*Ust))
    # #
    # #
    # # u = np.arange(U)
    # # uf0 = uf(u,Fmin,Ust)
    # #
    # # var = 100 #hertz square
    # #
    # # Wf0 = np.load('WF.npy')
    # #
    # # time = np.arange(time_bins)
    #
    # K = 10
    # P = 30
    #
    # #W_G = np.load('nmf1/Wgamma.npy')
    #
    # # H_G = np.random.rand(P,K)
    # H_phi = np.random.rand(K,time_bins)
    # # WM = np.random.rand(F,U)
    # HM = np.random.rand(U,time_bins)
    # Hf0 = np.random.rand(U,time_bins)

    npz_name = (path.split('/')[6]).split('.')[0] + '.npz'
    print(npz_name)
    np.savez(os.path.join(dest,npz_name), psd)

if __name__=='__main__':

    #path = '/mnt/data/datasets/Bach10_v1.1/01-AchGottundHerr/'
    path = '/mnt/data/datasets/Bach10_v1.1/01-AchGottundHerr/01-AchGottundHerr.wav'
    generate(path,dest = './test_stuffs/')
