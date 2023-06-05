import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve
from grad_forward import grad_forward

def osmosis_discretization(v):
    mx, my = v.shape

    x = np.linspace(1, mx, mx)
    y = np.linspace(1, my, my)
    hx = (max(x) - min(x)) / (mx - 1)
    hy = (max(y) - min(y)) / (my - 1)

    _, D1classic, D2classic = grad_forward(np.ones((mx, my)))

    Dxx_old = D1classic.T @ D1classic
    Dyy_old = D2classic.T @ D2classic

    m1xup = diags([1, 1], [0, 1], shape=(mx, mx)) / 2
    m1xlow = diags([1, 1], [-1, 0], shape=(mx, mx)) / 2
    m1yup = diags([1, 1], [0, 1], shape=(my, my)) / 2
    m1ylow = diags([1, 1], [-1, 0], shape=(my, my)) / 2

    M1xup = kron(eye(my), m1xup)
    M1xlow = kron(eye(my), m1xlow)
    M1yup = kron(m1yup, eye(mx))
    M1ylow = kron(m1ylow, eye(mx))

    d1ij = np.zeros((mx + 1, my))
    d2ij = np.zeros((mx, my + 1))

    d1ij[1:mx, :] = np.diff(v, axis=0) / (v[1:mx, :] + v[:mx - 1, :]) * 2 / hx
    d2ij[:, 1:my] = np.diff(v, axis=1) / (v[:, 1:my] + v[:, :my - 1]) * 2 / hy

    Ax = Dxx_old + 1 / hx * (diags(d1ij[1:mx + 1, :].flatten(), 0) @ M1xup - diags(d1ij[:mx, :].flatten(), 0) @ M1xlow)
    Ay = Dyy_old + 1 / hy * (diags(d2ij[:, 1:my + 1].flatten(), 0) @ M1yup - diags(d2ij[:, :my].flatten(), 0) @ M1ylow)

    A = Ax + Ay

    return A
