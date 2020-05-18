import pandas as pd
import numpy as np

def concatenate(lst, inp, i, name):
    out = lst
    if i == len(inp):
        return out
    else:
        a = pd.concat([lst, inp[i].rename(f'p{i+1}_{name}')], axis=1)
        i += 1
        return concatenate(a, inp, i, name)

def preprocess(data):
    a = []
    b = []
    x = pd.DataFrame(columns=['x'])

    if len(data) == 1:
        for k in range(0, 784, 28):
            a.append(data.iloc[:1, k:k+28].mean(axis=1))
            b.append(data.iloc[:1, k:k+28].std(axis=1))
    
    else:
        for k in range(0, 784, 28):
            a.append(data.iloc[:, k:k+28].mean(axis=1))
            b.append(data.iloc[:, k:k+28].std(axis=1))

    s = concatenate(x, a, 0, "mean")
    X = concatenate(s, b, 0, "std")

    X = X.drop('x', axis=1)
    X = X.iloc[:].values
    return np.interp(X, (X.min(), X.max()), (-1, +1))
