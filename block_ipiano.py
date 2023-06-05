import numpy as np
from scipy.sparse.linalg import bicgstab
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from grad_forward import *
from osmosis_discretization import *
from norms import *
from primal_dual import *

def huber(y, epsilon):
    t = norms(y, 2, 2)
    idx = t < epsilon
    z = t - epsilon / 2
    z[idx] = (t[idx] ** 2) / (2 * epsilon)
    return z


def block_ipiano(f, b, alpha, u, v, params):
    eta = params['eta']  # for the regulariser
    osm = params['osm']  # for osmosis
    mu = params['mu']  # for fidelity on v
    gamma = params['gamma']  # for fidelity on u
    lambda1 = params['lambda1']
    lambda2 = params['lambda2']
    beta1 = params['beta1']
    beta2 = params['beta2']
    L1 = params['L1']
    L2 = params['L2']
    N = params['N']
    T = params['T']
    tol_ipiano = params['tol_ipiano']
    tol_primal_dual = params['tol_primal_dual']
    flag_verbose = params['flag_verbose']

    mm, nn = u.shape

    _, D1, D2 = grad_forward(u)
    options = {'niter': T, 'tol': tol_primal_dual}

    K = lambda v: np.reshape(np.concatenate((D1 * v.flatten(), D2 * v.flatten())), (mm, nn, 2))
    KS = lambda y: np.reshape(D1.T @ y[:, :, 0].flatten() + D2.T @ y[:, :, 1].flatten(), (mm, nn))
    ProxFS = lambda y, sigma: (y / (1 + sigma * params['epsilon'])) / np.tile(np.maximum(1, np.linalg.norm(y / (1 + sigma * params['epsilon']), axis=(0,1)) / eta), (y.shape[0], y.shape[1], 1))

    GRAD = lambda  u: np.stack((np.reshape(D1 @ u.flatten(order='F'), (mm, nn), order='F'), np.reshape(D2 @ u.flatten(order='F'), (mm, nn), order='F')), axis=2)
    DIV = lambda u: np.reshape(D1.T @ np.reshape(u[:, :, 0], -1, order='F') + D2.T @ np.reshape(u[:, :, 1], -1, order='F'), (mm, nn), order='F')


    g1 = lambda v: (eta) * huber(GRAD(v), params['epsilon'])
    g2 = lambda u, v: (osm / 2) * v * np.power(norms(GRAD(u / v), 2, 2), 2)
    g3 = lambda v: (mu / 2) * np.power(norms(v[:, :, np.newaxis] - np.power(f[:, :, np.newaxis], alpha[:, :, np.newaxis]) * np.power(b[:, :, np.newaxis], (1 - alpha[:, :, np.newaxis])), 2, 2), 2)
    g4 = lambda u: (gamma / 2) * norms(np.sqrt(alpha[:, :, np.newaxis]) * (u[:, :, np.newaxis] - f[:, :, np.newaxis]), 2, 2) ** 2

    pixel_energy = lambda u, v: g1(v) + g2(u, v) + g3(v) + g4(u)
    total_energy = lambda u, v: np.sum(pixel_energy(u, v))

    Ouv = lambda u, v: g2(u, v) + g3(v)

    dvO_1 = lambda u, v: - np.sum(GRAD(u) * GRAD(u), axis=2) / (v ** 2)
    dvO_2 = lambda u, v: +(4 * u / v ** 3) * np.sum(GRAD(u) * GRAD(v), axis=2)
    dvO_3 = lambda u, v: - DIV(2 * (u / v ** 2)[:, :, np.newaxis] * GRAD(u))
    dvO_4 = lambda u, v: - 3 * u ** 2 / v ** 4 * np.sum(GRAD(v) ** 2, axis=2)
    dvO_5 = lambda u, v: +v * DIV((u ** 2 / v ** 3)[:, :, np.newaxis] * GRAD(v))

    grad_duO = lambda A, u, v: (osm / (v + 1e-10)) * np.reshape(A @ u.flatten(), (mm, nn))
    grad_dvO = lambda u, v: (osm / 2) * (dvO_1(u, v) + dvO_2(u, v) + dvO_3(u, v) + dvO_4(u, v) + dvO_5(u, v)) + mu * (v - (f ** alpha) * (b ** (1 - alpha)))

    test_u = lambda p1, p2, A, u, v, L: np.sum(Ouv(p1, v) - Ouv(u, v) - grad_duO(A, u, v) * (p1 - u) - (L / 2) * norms(p1[:, :, np.newaxis] - u[:, :, np.newaxis], 2, 2) ** 2 - 1e-10)
    test_v = lambda p1, p2, u, v, L: np.sum(Ouv(u, p2) - Ouv(u, v) - grad_dvO(u, v) * (p2 - v) - (L / 2) * norms(p2[:, :, np.newaxis] - v[:, :, np.newaxis], 2, 2) ** 2 - 1e-10)

    u_old = u
    v_old = v
    E = np.full(N, np.nan)

    if flag_verbose:
        print('  ITER   | (   L1    |    L2   ) |   energy   |   diff   |  PD Res (iter) ')
        print('-------------------------------------------------------------------------')

    for n in range(N):

        flag = 1

        A = osmosis_discretization(v)
        grad_u = grad_duO(A, u, v)
        grad_v = grad_dvO(u, v)

        flag_compute_u = 1
        flag_compute_v = 1

        while flag:

            if beta1 > 0.5:
                an1 = 1.99 * (1 - beta1) / L1
            else:
                an1 = 0.99 * (1 - 2 * beta1) / L1

            if beta2 > 0.5:
                an2 = 1.99 * (1 - beta2) / L2
            else:
                an2 = 0.99 * (1 - 2 * beta2) / L2

            if flag_compute_u:
                ud = u - an1 * grad_u + beta1 * (u - u_old)
            if flag_compute_v:
                vd = v - an2 * grad_v + beta2 * (v - v_old)

            ProxGu = lambda q, an: (gamma * alpha * f + ud / an) / (gamma * alpha + 1 / an)
            ProxGv = lambda q, tau: (vd / an2 + q / tau) / (1 / an2 + 1 / tau)

            if flag_compute_u:
                p1 = ProxGu(ud, an1)
            if flag_compute_v:
                p2, res, iter = primal_dual(vd, 1 / an2, K, KS, ProxFS, ProxGv, options)

            gap1 = test_u(p1, v, A, u, v, L1)
            gap2 = test_v(u, p2, u, v, L2)

            if gap1 < 0 and flag_compute_u:
                u_old = u
                u = p1
                L1 = L1 / lambda2
                flag_compute_u = 0
            else:
                if flag_compute_u:
                    L1 = lambda1 * L1

            if gap2 < 0 and flag_compute_v:
                v_old = v
                v = p2
                L2 = L2 / lambda2
                flag_compute_v = 0
            else:
                if flag_compute_v:
                    L2 = lambda1 * L2

            if flag_compute_u == 0 and flag_compute_v == 0:
                E[n] = total_energy(u, v)
                energy_diff = abs(total_energy(u, v) - total_energy(u_old, v_old)) / abs(total_energy(u_old, v_old))
                flag = 0
                if flag_verbose:
                    print(f'   {n:03d}   | ({L1:.2e} | {L2:.2e}) | {E[n]:.4e} | {energy_diff:.2e} | {res:.2e} ({iter:d})')

        if (energy_diff < tol_ipiano and n >= 5):
            break

    return u, v, L1, L2, E
