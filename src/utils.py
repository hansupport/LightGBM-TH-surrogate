import numpy as np

def nt2idx(n, t, T):
    return n * T + t

def idx2nt(idx, T):
    return idx // T, idx % T

def add_time_feature(x, T, P, TIME_DIM):
    # x = (N, F) -> x_new = (N*T, F+TIME_DIM) = (N*T, P)
    x_new = np.empty((x.shape[0] * T, P))
    for n in range(x.shape[0]):
        for t in range(T):
            x_feature = x[n] # (F, )
            x_time = [t ** i for i in range(1, TIME_DIM + 1)] # (TIME_DIM, )
            x_new[nt2idx(n, t, T)] = np.concatenate([x_feature, x_time]) # (F+TIME_DIM, ) = (P, )
    return x_new