import numpy as np
from scipy.linalg import dmft

M = 1000
N = 700

F = dft(M)
D = np.eye(M)[:,np.random.choice(M,N,replace=True)]
P = np.diag(np.exp(1j * np.random.uniform(-np.pi,np.pi, N)))

# This is the data matrix
A = F@D@P
