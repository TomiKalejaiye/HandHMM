from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class HMM(BaseEstimator,ClassifierMixin):

    """ An HMM classifier """

    def __init__(self, As=None, Bs=None, PI=None):
        self.As = As
        self.Bs = Bs
        self.PI = PI
    
    def fit(self,):
        num_states = 10
        #num_sym = len(set(O_movements))

        # Transition Probabilities
        As = {'cyl':np.array([]),'hook':np.array([]),'lat':np.array([]),'palm':np.array([]),'spher':np.array([]),'tip':np.array([])}
        for A in As:
            As[A] = np.random.random(size=(num_states,num_states))
            As[A] = As[A] / np.sum(As[A] , axis=1)


        # Emission Probabilities
        Bs = {'cyl':np.array([]),'hook':np.array([]),'lat':np.array([]),'palm':np.array([]),'spher':np.array([]),'tip':np.array([])}
        for B in Bs:
            Bs[B] = np.random.random(size=(num_states,num_sym))
            Bs[B] = Bs[B] / np.sum(Bs[B], axis=1).reshape((-1,1))

        # Equal Probabilities for the initial distribution
        PI= np.zeros(num_states)
        PI[0] = 1