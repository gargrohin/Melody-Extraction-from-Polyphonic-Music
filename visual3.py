import numpy as np
import os
import librosa


def ISDistortion(X,Y):
    """
    value = ISDistortion(X, Y)

    Returns the value of the Itakura-Saito (IS) divergence between
    matrix X and matrix Y. X and Y should be two NumPy arrays with
    same dimension.
    """
    return np.sum((-np.log(X / Y) + (X / Y) - 1))

def nmf(path, dest):


    stuff = np.load(path)

    K = 10
    P = 30

    Ust = 5
    F = 1103 #psd.shape[0], constant for constant sr
    Fmin = librosa.note_to_hz('C2')
    Fmax = librosa.note_to_hz('C7')

    U = int(np.log2(Fmax/Fmin) * 12*Ust +1)


    WF0 = np.load('WF.npy')
    WPHI_voc =np.load('voc_WPHI.npy')
    WPHI_bg = np.load('bg_WPHI.npy')
    SX = stuff['arr_3']
    time_bins = SX.shape[1]

    HPHI_voc = np.random.rand(K,time_bins)
    HPHI_bg = np.random.rand(K,time_bins)
    WM = np.random.rand(F,U)
    HF0_voc = np.random.rand(U,time_bins)
    HF0_bg = np.random.rand(U,time_bins)
    HM =  np.random.rand(U,time_bins)


    numberOfIterations = 40

    SPHI_bg = np.dot(WPHI_bg, HPHI_bg)
    SF0_bg = np.dot(WF0, HF0_bg)
    SPHI_voc = np.dot(WPHI_voc, HPHI_voc)
    SF0_voc = np.dot(WF0, HF0_voc)
    SM = np.dot(WM, HM)
    hatSX = SF0_voc * SPHI_voc + SF0_bg * SPHI_bg + SM


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


        # updating HF0_voc

        tempNumFbyN = (SPHI_voc * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = SPHI_voc / np.maximum(hatSX, eps)

        HF0_voc = np.maximum( HF0_voc * (np.dot(WF0.T, tempNumFbyN) / np.maximum(np.dot(WF0.T, tempDenFbyN), eps)), eps)

        SF0_voc = np.maximum(np.dot(WF0, HF0_voc),eps)

        hatSX = np.maximum(SF0_voc * SPHI_voc  + SM,eps)

        recoError[counterError] = ISDistortion(SX, hatSX)
        print("Reconstruction error after HF0_voc   : ", recoError[counterError])
        counterError += 1

        # updating HPHI_voc

        tempNumFbyN = (SF0_voc * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = SF0_voc / np.maximum(hatSX, eps)


        HPHI_voc = np.maximum( HPHI_voc * (np.dot(WPHI_voc.T, tempNumFbyN) / np.maximum(np.dot(WPHI_voc.T, tempDenFbyN), eps)), eps)
        SPHI_voc = np.maximum(np.dot(WPHI_voc, HPHI_voc), eps)
        hatSX = np.maximum(SF0_voc * SPHI_voc  + SM, eps)

        recoError[counterError] = ISDistortion(SX, hatSX)
        print("Reconstruction error after HPHI_voc  : ", recoError[counterError])
        counterError += 1

        # # updating HF0_bg
        #
        # tempNumFbyN = (SPHI_bg * SX) / np.maximum(hatSX ** 2, eps)
        # tempDenFbyN = SPHI_bg / np.maximum(hatSX, eps)
        #
        # HF0_bg = np.maximum(HF0_bg * (np.dot(WF0.T, tempNumFbyN) / np.maximum(np.dot(WF0.T, tempDenFbyN), eps)), eps)
        #
        # SF0_bg = np.maximum(np.dot(WF0, HF0_bg),eps)
        #
        # hatSX = np.maximum(SF0_voc * SPHI_voc + SF0_bg * SPHI_bg + SM,eps)
        #
        # recoError[counterError] = ISDistortion(SX, hatSX)
        # print("Reconstruction error after HF0_bg   : ", recoError[counterError])
        # counterError += 1
        #
        # # updating HPHI_bg
        #
        # tempNumFbyN = (SF0_bg * SX) / np.maximum(hatSX ** 2, eps)
        # tempDenFbyN = SF0_bg / np.maximum(hatSX, eps)
        #
        #
        # HPHI_bg = np.maximum( HPHI_bg * (np.dot(WPHI_bg.T, tempNumFbyN) / np.maximum(np.dot(WPHI_bg.T, tempDenFbyN), eps)), eps)
        # SPHI_bg = np.maximum(np.dot(WPHI_bg, HPHI_bg), eps)
        # hatSX = np.maximum(SF0_voc * SPHI_voc + SF0_bg * SPHI_bg + SM, eps)
        #
        # recoError[counterError] = ISDistortion(SX, hatSX)
        # print("Reconstruction error after HPHI_bg  : ", recoError[counterError])
        # counterError += 1

        # updating HM

        tempNumFbyN = SX / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = 1 / np.maximum(hatSX, eps)

        HM = np.maximum(HM * (np.dot(WM.T, tempNumFbyN) /np.maximum(np.dot(WM.T, tempDenFbyN), eps)) , eps)
        SM = np.maximum(np.dot(WM, HM), eps)
        hatSX = np.maximum(SF0_voc * SPHI_voc + SM, eps)

        recoError[counterError] = ISDistortion(SX, hatSX)
        print("Reconstruction error after HM        : ", recoError[counterError])
        counterError += 1

        # updating WM

        tempNumFbyN = SX / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = 1 / np.maximum(hatSX, eps)

        WM = np.maximum(WM * (np.dot(tempNumFbyN, HM.T) / np.maximum(np.dot(tempDenFbyN, HM.T), eps)), eps)

        SM = np.maximum(np.dot(WM, HM), eps)

        hatSX = np.maximum(SF0_voc * SPHI_voc + SM, eps)

        recoError[counterError] = ISDistortion(SX, hatSX)
        print("Reconstruction error after WM        : ",recoError[counterError])
        counterError += 1


        if recoError[counterError-1] < min_error:
            np.savez(dest,HPHI_bg,HPHI_voc, HF0_bg, HF0_voc, WM, HM, hatSX)
            min_error = recoError[counterError-1]
            print("out->")
        else:
            break

def nmf_multi(path,dest = './outs_voc/'):

    # fh = open('outs.txt','r')
    # done_list = [x.strip()[4:] for x in fh]
    # print(done_list)
    s=0
    for song in os.listdir(path):
        flag = False
        print()
        print(song)
        print()
        # if song in done_list:
        #     print("allready done")
        #     print("")
        #     continue

        song_path = os.path.join(path,song)
        stuff = np.load(song_path)



        K = 10
        P = 30

        Ust = 5
        F = 1103 #psd.shape[0], constant for constant sr
        Fmin = librosa.note_to_hz('C2')
        Fmax = librosa.note_to_hz('C7')

        U = int(np.log2(Fmax/Fmin) * 12*Ust +1)

        eps = 10 ** (-20)


        WF0 = np.load('WF.npy')
        WGAMMA = np.load('Wgamma.npy')
        HGAMMA= np.random.rand(P,K)
        SX = stuff['arr_0']
        time_bins = SX.shape[1]

        HPHI = np.random.rand(K,time_bins)
        WM = np.random.rand(F,U)
        HF0 = np.random.rand(U,time_bins)
        HM =  np.random.rand(U,time_bins)

        if s > 0:
            stuff = np.load(output_path)
            HGAMMA = stuff['arr_2']
        else:
            s+=1

        numberOfIterations = 100

        WPHI = np.maximum(np.dot(WGAMMA, HGAMMA),eps)
        SPHI = np.maximum(np.dot(WPHI, HPHI),eps)
        SF0 = np.maximum(np.dot(WF0, HF0) ,eps)
        SM = np.maximum(np.dot(WM, HM), eps)
        hatSX = np.maximum(SF0 * SPHI + SM , eps)

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

        output_path = dest + 'out_' + song


        for n in np.arange(numberOfIterations):
            print("iteration ", n, " over ", numberOfIterations)

            error_IS[n] = ISDistortion(SX,hatSX)


            # updating HF0

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


            if flag:
                if n == 15:
                    np.savez(output_path,WF0,WGAMMA,HGAMMA,HPHI, WM, HF0, HM, hatSX)
                    print("out->")
                    break

            if recoError[counterError-1] < min_error:
                min_error = recoError[counterError -1]
                np.savez(output_path,WF0,WGAMMA,HGAMMA,HPHI, WM, HF0, HM, hatSX)
                print("out->")
            else:
                if n < 3:
                    flag = True
                    continue
                if not flag:
                    break
    np.save('outs_voc/voc_Hgamma.npy' , HGAMMA)



if __name__ == '__main__':
    np.random.seed(0)
    path = './stuffs_voc/'
    dest = './outs_v/LizNelson_Rainfall_v.npz'

    #nmf_multi(path)
    nmf('./stuffs/LizNelson_Rainfall.npz', dest)
