import pandas as pd
import numpy as np
import librosa
from scipy.io import loadmat
import os

'''
The function `make_ground` creates the ground truth (midi like) file for each audio file.

The functions `RCA` and `RPA` calculate the Raw Chroma Accuracy an the Raw Pitch Accuracy using the output of NMF and the ground truth. These functions are not used in the `make_groud` function, rather are imported separately in a jupyter notebook.
'''

def RCA():
    n=N
    acc = 0
    for i in range(N):
        min = np.max
        pr = np.nonzero(predict[:,i])[0]

        if gl[i] == 0:
            n-=1
            continue
        if pr.size==0:
            continue

        x = np.min(abs(pr - gl[i]))

        if x == 0:
            acc+=1
    return acc/n

def RPA():
    n=N
    acc = 0
    for i in range(N):
        min = np.max
        pr = np.nonzero(predict[:,i])[0]

        if gl[i] == 0:
            n-=1
            continue
        if pr.size==0:
            continue

        x = np.min(abs(pr - gl[i]))

        if x == 0:
            acc+=1
    return acc/n

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def uf(u,Fmin,Ust):
    return Fmin * np.power(2,(u)/(12*Ust))

def make_ground():

    # ground_path = '/mnt/Stuff/Acads/UGP/mycode/ground_voc'
    # ground_list = [x.split('.')[0].strip() for x in os.listdir(ground_path)]

    #os.chdir("./ground_voc")

    Fmin = librosa.note_to_hz('C2')
    Fmax = librosa.note_to_hz('C7')

    note = uf(np.arange(int(np.log2(Fmax/Fmin) * 12*1 +1)) , Fmin,1)

    # melody2 = "/mnt/Stuff/Acads/UGP/medleydb/medleydb/data/Annotations/Melody/\
    #             Melody2/"

    # mirk = "/mnt/data/datasets/MIR-1K/"

    bach = "/mnt/data/datasets/Bach10_v1.1/"

    # for song in os.listdir(melody2):
    #     print(song.rsplit(".")[0][:-8])
    #     path = melody2 + song
    #     liz = pd.read_csv(path,names = ['time','freq'])
    #     liz = liz.to_numpy()
    #
    #     #print(liz.shape[0])
    #
    #     N = int(liz.shape[0]/2 +1)
    #     #print(N)
    #
    #     i=0
    #     #ground_liz = np.zeros([61,N])
    #     gl = np.zeros(N)
    #     for x in liz:
    #         if i%2==0:
    #             gl[int(i/2)] = np.argwhere(note == find_nearest(note,x[1]))
    #         #ground_liz[int(gl[int(i/2)]) , int(i/2)] = 1
    #         i+=1
    #     save_path = song.rsplit(".")[0][:-8] + ".npy"
    #     np.save(save_path , gl)
    #     print(" Done.")

    # for file in os.listdir(mirk):
    #     if file.split('.')[1] == 'pv':
    #         path = mirk + file
    #         liz = []
    #         ff = open(path , 'r')
    #         freq = [float(x) for x in ff]
    #
    #         N = len(freq)
    #
    #         print(N)
    #
    #         i = 0
    #         gl = np.zeros(N)
    #         for x in freq:
    #             gl[i] = round(x) #np.argwhere(note == find_nearest(note , x))
    #             i+=1
    #         save_path = file.split('.')[0] + '.npy'
    #         np.save(save_path , gl)
    #         print('Done.')

    for song in os.listdir(bach):
        print(song)
        file = bach + song + '/' + song + '-GTF0s.mat'
        f = loadmat(file)
        f = f['GTF0s']
        i=0
        for ch in ['violin', 'clarinet','saxophone','bassoon']:
            fr = f[i]
            i+=1
            N = fr.shape[0]
            gl = np.zeros(N)
            j=0
            for x in fr:
                gl[j] = x - 36 +1
                j+=1
            save_path = 'ground_bach/' + song  + '-' + ch + '.npy'
            np.save(save_path , gl)
            print('Done.')



if __name__ == "__main__":
    make_ground()
