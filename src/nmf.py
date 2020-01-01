import numpy as np
import os
import librosa

'''
	nmf function: implementation of Main Melody Extraction With Source-Filter NMF AND CRNN (Basaran et al, 2018). Uses the Non Negative matrix decomposition algorithm given in Algorithms for Non-negative Matrix
Factorization (Lee et al).

The input of the function is the path to the preprocessed data for one data point.
It also loads the pre-defined WF0 and WGamma found in data folder, created in one of the notebooks. The path to them needs to be changed.

The output are the rest of the matrices after nmf decomposition. The can be seen at the end of the function.

'''

def ISDistortion(X,Y):
    """
    value = ISDistortion(X, Y)

    Returns the value of the Itakura-Saito (IS) divergence between
    matrix X and matrix Y. X and Y should be two NumPy arrays with
    same dimension.
    """
    return np.sum((-np.log(X / Y) + (X / Y) - 1))

def nmf(path):


    stuff = np.load(path)

    K = 10
    P = 30

    Ust = 5
    F = 1103 #psd.shape[0], constant for constant sr
    Fmin = librosa.note_to_hz('C2')
    Fmax = librosa.note_to_hz('C7')

    U = int(np.log2(Fmax/Fmin) * 12*Ust +1)


    WF0 = np.load('WF.npy')
    WGAMMA = np.load('Wgamma.npy')
    HGAMMA= np.random.rand(P,K)
    SX = stuff['arr_0']
    time_bins = SX.shape[1]

    HPHI = np.random.rand(K,time_bins)
    WM = np.random.rand(F,U)
    HF0 = np.random.rand(U,time_bins)
    HM =  np.random.rand(U,time_bins)


    numberOfIterations = 20

    WPHI = np.dot(WGAMMA, HGAMMA)
    SPHI = np.dot(WPHI, HPHI)
    SF0 = np.dot(WF0, HF0)
    SM = np.dot(WM, HM)
    hatSX = SF0 * SPHI + SM
    # for i in hatSX:
    #     for j in i:
    #         if j == 0.0:
    #             print('wtf\n')

    #SX = np.ones([hatSX.shape[0],hatSX.shape[1]])
    #print(SX)

    NF0 = WF0.shape[1]
    F = SX.shape[0]
    N = SX.shape[1]

    # temporary matrices
    tempNumFbyN = np.zeros([F, N])
    tempDenFbyN = np.zeros([F, N])


    min_error = 10**20
    recoError = np.zeros([numberOfIterations * 5 * 2 + NF0 * 2 + 1])
    recoError[0] = ISDistortion(SX, hatSX)

    print("Reconstruction error at beginning: ", recoError[0])

    counterError = 1

    error_IS = np.zeros(numberOfIterations)

    eps = 10 ** (-20)

    for n in np.arange(numberOfIterations):
        print("iteration ", n, " over ", numberOfIterations)

        error_IS[n] = ISDistortion(SX,hatSX)


        # updating Hf0

        tempNumFbyN = (SPHI * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = SPHI / np.maximum(hatSX, eps)

        HF0 = HF0 * (np.dot(WF0.T, tempNumFbyN) / np.maximum(np.dot(WF0.T, tempDenFbyN), eps))

        SF0 = np.maximum(np.dot(WF0, HF0),eps)

        hatSX = np.maximum(SF0 * SPHI + SM,eps)

        recoError[counterError] = ISDistortion(SX, hatSX)
        print("Reconstruction error after HF0   : ", recoError[counterError])
        counterError += 1

        tempNumFbyN = (SF0 * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = SF0 / np.maximum(hatSX, eps)


        HPHI = HPHI * (np.dot(WPHI.T, tempNumFbyN) / np.maximum(np.dot(WPHI.T, tempDenFbyN), eps))
        SPHI = np.maximum(np.dot(WPHI, HPHI), eps)
        hatSX = np.maximum(SF0 * SPHI + SM, eps)

        recoError[counterError] = ISDistortion(SX, hatSX)
        print("Reconstruction error after HPHI  : ", recoError[counterError])
        counterError += 1

        # updating HM
        tempNumFbyN = SX / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = 1 / np.maximum(hatSX, eps)

        HM = np.maximum(HM * (np.dot(WM.T, tempNumFbyN) /np.maximum(np.dot(WM.T, tempDenFbyN), eps)) , eps)
        SM = np.maximum(np.dot(WM, HM), eps)
        hatSX = np.maximum(SF0 * SPHI + SM, eps)

        recoError[counterError] = ISDistortion(SX, hatSX)
        print("Reconstruction error after HM    : ", recoError[counterError])
        counterError += 1

        # updating HGAMMA
        tempNumFbyN = (SF0 * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = SF0 / np.maximum(hatSX, eps)

        HGAMMA = np.maximum( HGAMMA * (np.dot(WGAMMA.T, np.dot(tempNumFbyN, HPHI.T)) / np.maximum(np.dot(WGAMMA.T, np.dot(tempDenFbyN, HPHI.T)),
                                                                                                  eps)), eps)

        WPHI = np.maximum(np.dot(WGAMMA, HGAMMA), eps)
        SPHI = np.maximum(np.dot(WPHI, HPHI), eps)
        hatSX = np.maximum(SF0 * SPHI + SM, eps)

        recoError[counterError] = ISDistortion(SX, hatSX)
        print("Reconstruction error after HGAMMA: ",recoError[counterError])
        counterError += 1

        # updating WM

        tempNumFbyN = SX / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = 1 / np.maximum(hatSX, eps)

        WM = np.maximum(WM * (np.dot(tempNumFbyN, HM.T) / np.maximum(np.dot(tempDenFbyN, HM.T), eps)), eps)

        SM = np.maximum(np.dot(WM, HM), eps)

        hatSX = np.maximum(SF0 * SPHI + SM, eps)

        recoError[counterError] = ISDistortion(SX, hatSX)
        print("Reconstruction error after WM    : ",recoError[counterError])
        counterError += 1


        # if recoError[counterError-1] < min_error:
        #     np.savez('out_amy_1.npz',WF0,WGAMMA,HGAMMA,HPHI, WM, HF0, HM, hatSX)
        #     min_error = recoError[counterError-1]
        #     print("out->")
        # else:
        #     break
    np.savez('out_06-DieSonne.npz.npz',WF0,WGAMMA,HGAMMA,HPHI, WM, HF0, HM, hatSX)


def nmf_multi(path,dest = './outs_nonsax/'):

    # fh = open('outs.txt','r')
    # done_list = [x.strip()[4:] for x in fh]
    # print(done_list)
    s=0
    WF0 = np.load('WF.npy')
    WGAMMA = np.load('Wgamma.npy')
    epoch = 100
    while(epoch):
        print(epoch)
        print()
        epoch-=1
        for song in os.listdir(path):
            flag = False
            # if 'leon' not in song:
            #     continue
            print()
            print(song)
            print()
            # if song in done_list:
            #     print("allready done")
            #     print("")
            #     continue

            song_path = os.path.join(path,song)
            stuff = np.load(song_path)

            output_path = dest + 'out_' + song

            K = 10
            P = 30

            numberOfIterations = 1

            Ust = 5
            F = 1103 #psd.shape[0], constant for constant sr
            Fmin = librosa.note_to_hz('C2')
            Fmax = librosa.note_to_hz('C7')

            U = int(np.log2(Fmax/Fmin) * 12*Ust +1)

            eps = 10 ** (-20)

            SX = stuff['arr_0']
            time_bins = SX.shape[1]

            HPHI = np.random.rand(K,time_bins)
            WM = np.random.rand(F,U)
            HF0 = np.random.rand(U,time_bins)
            HM =  np.random.rand(U,time_bins)

            if epoch < 99:
                stuff = np.load(output_path)
                HPHI = stuff['arr_1']
                WM = stuff['arr_2']
                HF0 = stuff['arr_3']
                HM = stuff['arr_4']

            if s>0:
                HGAMMA = np.load('outs_nonsax/Hgamma_nonsax.npy')
            else:
                s+=1
                HGAMMA= np.random.rand(P,K)


            WPHI = np.maximum(np.dot(WGAMMA, HGAMMA),eps)
            SPHI = np.maximum(np.dot(WPHI, HPHI),eps)
            SF0 = np.maximum(np.dot(WF0, HF0) ,eps)
            SM = np.maximum(np.dot(WM, HM), eps)
            hatSX = np.maximum(SF0 * SPHI + SM , eps)
            prev = WPHI

            NF0 = WF0.shape[1]
            F = SX.shape[0]
            N = SX.shape[1]

            # temporary matrices
            tempNumFbyN = np.zeros([F, N])
            tempDenFbyN = np.zeros([F, N])


            min_error = 10**20
            #recoError = np.zeros([numberOfIterations * 5 * 2 + NF0 * 2 + 1])
            #recoError[0] = ISDistortion(SX, hatSX)

            #print("Reconstruction error at beginning: ", recoError[0])

            counterError = 1

            error_IS = np.zeros(numberOfIterations)


            for n in np.arange(numberOfIterations):
                print("iteration ", n, " over ", numberOfIterations)

                error_IS[n] = ISDistortion(SX,hatSX)


                # updating HF0

                tempNumFbyN = (SPHI * SX) / np.maximum(hatSX ** 2, eps)
                tempDenFbyN = SPHI / np.maximum(hatSX, eps)

                HF0 = HF0 * (np.dot(WF0.T, tempNumFbyN) / np.maximum(np.dot(WF0.T, tempDenFbyN), eps))

                SF0 = np.maximum(np.dot(WF0, HF0),eps)

                hatSX = np.maximum(SF0 * SPHI + SM,eps)

                # recoError[counterError] = ISDistortion(SX, hatSX)
                # print("Reconstruction error after HF0   : ", recoError[counterError])
                # counterError += 1

                tempNumFbyN = (SF0 * SX) / np.maximum(hatSX ** 2, eps)
                tempDenFbyN = SF0 / np.maximum(hatSX, eps)


                HPHI = HPHI * (np.dot(WPHI.T, tempNumFbyN) / np.maximum(np.dot(WPHI.T, tempDenFbyN), eps))
                SPHI = np.maximum(np.dot(WPHI, HPHI), eps)
                hatSX = np.maximum(SF0 * SPHI + SM, eps)

                # recoError[counterError] = ISDistortion(SX, hatSX)
                # print("Reconstruction error after HPHI  : ", recoError[counterError])
                # counterError += 1

                # updating HM
                tempNumFbyN = SX / np.maximum(hatSX ** 2, eps)
                tempDenFbyN = 1 / np.maximum(hatSX, eps)

                HM = np.maximum(HM * (np.dot(WM.T, tempNumFbyN) /np.maximum(np.dot(WM.T, tempDenFbyN), eps)) , eps)
                SM = np.maximum(np.dot(WM, HM), eps)
                hatSX = np.maximum(SF0 * SPHI + SM, eps)

                # recoError[counterError] = ISDistortion(SX, hatSX)
                # print("Reconstruction error after HM    : ", recoError[counterError])
                # counterError += 1

                # updating HGAMMA
                tempNumFbyN = (SF0 * SX) / np.maximum(hatSX ** 2, eps)
                tempDenFbyN = SF0 / np.maximum(hatSX, eps)

                HGAMMA = np.maximum( HGAMMA * (np.dot(WGAMMA.T, np.dot(tempNumFbyN, HPHI.T)) / np.maximum(np.dot(WGAMMA.T, np.dot(tempDenFbyN, HPHI.T)),
                                                                                                          eps)), eps)

                WPHI = np.maximum(np.dot(WGAMMA, HGAMMA), eps)
                SPHI = np.maximum(np.dot(WPHI, HPHI), eps)
                hatSX = np.maximum(SF0 * SPHI + SM, eps)

                # recoError[counterError] = ISDistortion(SX, hatSX)
                # print("Reconstruction error after HGAMMA: ",recoError[counterError])
                # counterError += 1

                # updating WM

                tempNumFbyN = SX / np.maximum(hatSX ** 2, eps)
                tempDenFbyN = 1 / np.maximum(hatSX, eps)

                WM = np.maximum(WM * (np.dot(tempNumFbyN, HM.T) / np.maximum(np.dot(tempDenFbyN, HM.T), eps)), eps)

                SM = np.maximum(np.dot(WM, HM), eps)

                hatSX = np.maximum(SF0 * SPHI + SM, eps)

                # recoError[counterError] = ISDistortion(SX, hatSX)
                # print("Reconstruction error after WM    : ",recoError[counterError])
                # counterError += 1

                # if flag:
                #     if n == 20:
                #         np.savez(output_path,WF0,WGAMMA,HGAMMA,HPHI, WM, HF0, HM, hatSX)
                #         print("out->")
                #         break
                #
                # if recoError[counterError-1] < min_error:
                #     min_error = recoError[counterError -1]
                #     np.savez(output_path,HGAMMA,HPHI, WM, HF0, HM, hatSX)
                #     print("out->")
                # else:
                #     if n < 10:
                #         flag = True
                #         continue
                #     if not flag:
                #         break

                if n == 20:
                    break

            np.savez(output_path,HGAMMA,HPHI, WM, HF0, HM, hatSX)
            print('out->')

            np.save('outs_nonsax/Hgamma_nonsax.npy' , HGAMMA)
            print()
            print('IS_Divergence WPHI: ' , ISDistortion(prev,WPHI))
            print()



if __name__ == '__main__':
    #np.random.seed(0)
    #path ='/data/rharish/stuffs_nonsax/'

    #nmf_multi(path)
    nmf('./06-DieSonne.npz')
