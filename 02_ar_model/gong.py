import numpy as np
import scipy.io.wavfile as wav
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def ar_fit_model(y: np.ndarray, p: int) -> np.ndarray:
    """Computes the parameters of the autogression model

    Args:
    -----
        y: np.ndarray: sound signal
        p: int: required order of AR model

    Return:
    -------
        np.ndarray: estimated parameters of AR model 

    Shape:
    ------
       - Input: (N,)
       - Output: (p+1,)
    """

    M = np.ones((y.shape[0]-p, p+1))
    for i in range(p, y.shape[0]):
        M[i-p,1:] = y[i-p:i]

    a = np.linalg.lstsq(M, y[p:])[0]
    return a

def ar_predict(a: np.ndarray, y0: np.ndarray, N:int) -> np.ndarray:
    """ computes the rest of elements of y, starting from (p+1)-th 
        one up to N-th one. 

    Args:
    -----
        a: np.ndarray: estimated parameters of AR model
        y0: np.ndarray: beginning of sequence to be predicted
        N: int:  required length of predicted sequence, including the 
                 beginning represented by y0.
    
    Return:
    -------
        np.ndarray: the predicted sequence 

    Shape:
    ------
       - Input: (p+1,), (p,)
       - Output: (N,)
    """
    y_pred = np.zeros(N)
    p = y0.shape[0]
    y_pred[:p] = y0

    for i in range(p, N):
        x = np.concatenate((np.array([1]), y_pred[i-p:i]))
        y_pred[i] = a.T @ x

    return y_pred

if(__name__ == '__main__'):

    fs,y=wav.read('data/gong.wav')
    y = y.copy()/32767
    p = 300      # size of history considered for prediction
    N = len(y)   # length of the sequence
    K = 10000    # visualize just K first elements

    a = ar_fit_model(y, p)

    y0 = y[:p]
    y_pred = ar_predict(a, y0, N)

    wav.write('data/gong_predicted.wav', fs, y_pred)

    plt.plot(y[:K], 'b', label = 'original')
    plt.plot(y_pred[:K], 'r', label = 'AR model')
    plt.legend()
    plt.show()
