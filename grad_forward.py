from scipy.sparse import diags, kron, eye
import numpy as np

def grad_forward(u):
    M, N = u.shape

    D1 = diags([-np.ones(M), np.ones(M)], [0, 1], shape=(M, M)).toarray()
    D2 = diags([-np.ones(N), np.ones(N)], [0, 1], shape=(N, N)).toarray()

    # Boundary conditions
    D1[-1, :] = 0
    D2[-1, :] = 0

    D1 = kron(eye(N), D1)
    D2 = kron(D2, eye(M))

    DU = np.zeros((M, N, 2))
    DU[:, :, 0] = (D1 @ u.flatten()).reshape(M, N)
    DU[:, :, 1] = (D2 @ u.flatten()).reshape(M, N)

    return DU, D1, D2
