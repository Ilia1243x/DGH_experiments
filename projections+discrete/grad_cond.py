import numpy as np
from functools import partial

from mappings import proj_simplex
from spaces import arrange_distances

class LearningRate:
    def __init__(self, lambda_ = 1e-3, s0 = 1, p = 0.5):
        self.lambda_: float = lambda_
        self.s0: float =s0
        self.p: float = p
        self.iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p
    
def dot_multiplicand(S,c,X,Y):
        X__Y, Y__X, _Y_X = arrange_distances(X, Y)
        c_Y_X, c__Y_X = c**_Y_X,  c**-_Y_X

        return (c__Y_X @ S @ c_Y_X + c_Y_X @ S @ c__Y_X).T + \
            c**-X__Y @ S @ c**Y__X + c**X__Y @ S @ c**-Y__X

def grad(S,c,X,Y):
        return 2 * dot_multiplicand(S,c,X,Y)

def gradient_descent(S_prev, max_iter, c, X, Y, tol=1e-10,lambda_: float = 1e-3):
    lr = LearningRate(lambda_=lambda_)
    n, m = len(X), len(Y)
    S = S_prev.copy()
    for iter in range(max_iter):
          S -= lr() * grad(S, c, X,Y)
          S = proj_simplex(S, n, m)
          if np.linalg.norm(S-S_prev, ord = 'fro') < tol:
                print(iter+1, c)
                return S, iter + 1
          S_prev = S.copy()
    return S, iter + 1

def Adam(S_prev, max_iter, c, X, Y, tol=1e-10,lambda_: float = 1e-3):
    lr = LearningRate(lambda_=lambda_)
    n, m = len(X), len(Y)
    S = S_prev.copy()
    eps: float = 1e-8
    beta_1: float = 0.9
    beta_2: float = 0.999
    m_t = np.zeros_like(S)
    v_t =  np.zeros_like(S)

    for iter in range(1, max_iter + 1):
        g_t = grad(S, c, X,Y)
        np.add(beta_1 * m_t, (1-beta_1) * g_t, out = m_t)
        np.add(beta_2 * v_t, (1-beta_2) * (g_t ** 2), out = v_t)
        m_hat = m_t / (1 - beta_1 ** iter)
        v_hat = v_t / (1 - beta_2 ** iter)
        S -= (lr() * m_hat / (np.sqrt(v_hat) + eps))
        S = proj_simplex(S, n, m)
        #if np.linalg.norm(S-S_prev, ord = 'fro') < tol:
        #    print(iter+1, c)
        #    return S, iter
        S_prev = S.copy()
    return S, iter
      