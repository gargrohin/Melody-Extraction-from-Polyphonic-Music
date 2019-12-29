import numpy as np

def ISDistortion(X,Y):
    """
    value = ISDistortion(X, Y)

    Returns the value of the Itakura-Saito (IS) divergence between
    matrix X and matrix Y. X and Y should be two NumPy arrays with
    same dimension.
    """
    return np.sum((-np.log(X / Y) + (X / Y) - 1))




stuff = np.load('stuff_liz.npz')


WF0 = stuff['arr_0']
WGAMMA = stuff['arr_1']
HGAMMA = stuff['arr_2']
HPHI = stuff['arr_3']
WM = stuff['arr_4']
HF0 = stuff['arr_5']
HM = stuff['arr_6']
SX = stuff['arr_7']


numberOfIterations = 50

WPHI = np.dot(WGAMMA, HGAMMA)
SPHI = np.dot(WPHI, HPHI)
SF0 = np.dot(WF0, HF0)
SM = np.dot(WM, HM)
hatSX = SF0 * SPHI + SM

NF0 = WF0.shape[1]
F = SX.shape[0]
N = SX.shape[1]

# temporary matrices
tempNumFbyN = np.zeros([F, N])
tempDenFbyN = np.zeros([F, N])



recoError = np.zeros([numberOfIterations * 5 * 2 + NF0 * 2 + 1])
recoError[0] = ISDistortion(SX, hatSX)

print("Reconstruction error at beginning: ", recoError[0])

counterError = 1

error_IS = np.zeros(numberOfIterations)

eps = 10 ** (-20)

for n in np.arange(numberOfIterations):
    print("iteration ", n, " over ", numberOfIterations)
    
    error_IS[n] = ISDistortion(SX,hatSX)

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
    
    

np.savez('out_liz.npz',WF0,WGAMMA,HGAMMA,HPHI, WM, HF0, HM, hatSX)