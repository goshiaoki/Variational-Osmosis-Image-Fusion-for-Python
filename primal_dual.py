import numpy as np

def getoptions(options, name, default_value):
    return options.get(name, default_value)

def primal_dual(x, mu, K, KS, ProxFS, ProxG, options):
    niter = getoptions(options, 'niter', 100)
    theta = getoptions(options, 'theta', 1)
    tol = getoptions(options, 'tol', 1e-2)
    verbose = getoptions(options, 'verbose', 0)
    acceleration = getoptions(options, 'acceleration', 1)

    # OPERATOR NORM
    L = 8

    if acceleration:
        # ADMM
        tau = 0.99 / L
        sigma = 0.99 / (tau * L)
        gamma = 0.5 * mu
    else:
        sigma = 10
        tau = 0.9 / (sigma * L)

    xhat = x
    y = K(x)
    xstar = KS(y)
    res = np.zeros(niter)

    for iter in range(niter):
        x_old = x
        y_old = y
        Kx_old = K(x)  # Kx_hat
        xstar_old = xstar  # KS(y)

        # DUAL PROBLEM
        Kx_hat = K(xhat)
        y = ProxFS(y + sigma * Kx_hat, sigma)

        # PRIMAL PROBLEM
        xstar = KS(y)
        x = ProxG(x - tau * xstar, tau)

        # EXTRAPOLATION
        xhat = x + theta * (x - x_old)

        # ACCELERATION
        if acceleration:
            theta = 1. / np.sqrt(1 + 2 * gamma * tau)
            tau = theta * tau
            sigma = sigma / theta

        # primal residual
        p_res = (x_old - x) / tau - (xstar_old - xstar)
        p = np.sum(np.abs(p_res))
        # dual residual
        d_res = (y_old - y) / sigma - (Kx_old - Kx_hat)
        d = np.sum(np.abs(d_res))

        res[iter] = (p + d) / np.size(x)

        if verbose:
            print('{}: res: {:.2e}'.format(iter, res[iter]))

        if res[iter] < tol:
            break

    res_final = res[iter]

    return x, res_final, iter
