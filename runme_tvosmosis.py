import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import datetime
import matplotlib.pyplot as plt
from TV_osmosis_wrapper import *
import time
import scipy.io
np.set_printoptions(suppress=True)


# Parameters
params = {}
params["plot_figures"] = False
params["flag_verbose"] = True

# for the model
ETA = [1]     #[0, 0.1, 0.5, 5], [0, 5, 10, 100]; # stack of weights for TV
MU = [100]      #[10, 100], [5, 10, 100]; # stack of weights for fidelity v
GAMMA = [0.1]     #[0, 1, 10], [1, 10, 100]; # stack of weights for fidelity u
EPSILON = [0.05] # HUBER-TV
# for iPiano (Alg. 4)
params["lambda1"] = 2   # backtracking update
params["lambda2"] = 1.2 # backtracking update
params["beta1"]   = 0.4
params["beta2"]   = 0.4
params["L1"]      = 1.0 # estimated starting Lipschitz
params["L2"]      = 1.0 # estimated starting Lipschitz
params["beta1"]   = 0.4 # inertial parameter
params["beta2"]   = 0.4 # inertial parameter
# for blurring alpha-map
FLAG_BLUR = [0] # 0 or 1 or [0,1]
BLUR = [5]      # [0.5,1.5,3,15]

# Initialisation
# 0: (u0 = f), 1: (u0 = alpha*f + (1-alpha)*b ), 2: AVG
FLAG_initialisation = [0]

# iterations
params["N"] = 10000 # maxiter iPiano
params["T"] = 10000 # maxiter primal-dual nested in iPiano4
# tolerances
params["tol_ipiano"] = 1e-6
params["tol_primal_dual"] = 1e-4

# START OF THE ALGORITHM
EXPERIMENT = [1] # here 1 = puppets, 4 = facefusion

for experiment in EXPERIMENT:
    for flag_blur in FLAG_BLUR:
        for blur_test in range(0, flag_blur * len(BLUR)  + 1): #for blur_test in range(1, flag_blur * (len(BLUR) - 1) + 1):
            for gamma in GAMMA:
                for mu in MU:
                    OSM = [1] # weights for osmosis: here fixed but it can be a stack
                    for osm in OSM: # which fidelity term for osmosis?
                        for eta in ETA:
                            for epsilon in EPSILON:
                                for flag_initialisation in FLAG_initialisation:

                                    params.update({
                                        'eta': eta,
                                        'osm': osm,
                                        'mu': mu,
                                        'gamma': gamma,
                                        'epsilon': epsilon,
                                        'flag_blur': flag_blur,
                                        'blur_test': blur_test,
                                        'flag_initialisation': flag_initialisation,
                                        'experiment': experiment
                                    })

                                    filename_u = f"./results/{experiment}_output_u_init{params['flag_initialisation']}_alphablur{params['flag_blur'] * BLUR[blur_test]}_eta{params['eta']}_mu{params['mu']}_gamma{params['gamma']}_eps{params['epsilon']}"
                                    filename_v = f"./results/{experiment}_output_v_init{params['flag_initialisation']}_alphablur{params['flag_blur'] * BLUR[blur_test]}_eta{params['eta']}_mu{params['mu']}_gamma{params['gamma']}_eps{params['epsilon']}"
                                    filename_txt = filename_u + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.txt'
                                    params['fileID'] = filename_txt

                                    with open(filename_txt, 'w') as file:
                                        # Load images
                                        foreground = np.array(Image.open(f"dataset/{experiment}_foreground.png")).astype(np.double) / 255.0
                                        background = np.array(Image.open(f"dataset/{experiment}_background.png")).astype(np.double) / 255.0
                                        alpha = np.array(Image.open(f"dataset/{experiment}_alpha.png")).astype(np.double) / 255.0

                                        if params['flag_blur']:
                                            alpha = gaussian_filter(alpha == 1, BLUR[blur_test])

                                        if max(foreground.shape[0], foreground.shape[1]) > 1500 and experiment == 2:
                                            # Use suitable function in Python for imresize_old
                                            foreground = imresize_old(foreground, 0.4)
                                            background = imresize_old(background, 0.4)
                                            alpha = imresize_old(alpha, 0.4)

                                        C = foreground.shape[2]

                                        ufinal = np.zeros_like(foreground)
                                        vfinal = np.zeros_like(foreground)
                                        E = np.full((params['N'], foreground.shape[2]), np.nan)

                                        # Print to file
                                        print('\n########################')
                                        print('###### TV-OSMOSIS ######')
                                        print('########################\n')

                                        print(f'experiment: {experiment}')

                                        if params['flag_blur']:
                                            print(f'alpha blur: {BLUR[blur_test]}')
                                        else:
                                            print('alpha blur: no')

                                        if params['flag_initialisation'] == 0:
                                            print('initial.  : u0 = f')
                                        elif params['flag_initialisation'] == 1:
                                            print('initial.  : u0 = alpha*f + (1-alpha)*b')
                                        elif params['flag_initialisation'] == 2:
                                            print('initial.  : u0 = alpha*avg(f) + (1-alpha)*avg(b)')

                                        print('\n###### PARAMETERS ######')
                                        print('-- model --')
                                        print(f'eta             : {params["eta"]:.2f}')
                                        print(f'mu              : {params["mu"]:.2f}')
                                        print(f'gamma           : {params["gamma"]:.2f}')
                                        print(f'epsilon (huber) : {params["epsilon"]:.2f}')
                                        print('-- iPiano --')
                                        print(f'beta1           : {params["beta1"]:.2f}')
                                        print(f'beta2           : {params["beta2"]:.2f}')
                                        print(f'lambda1         : {params["lambda1"]:.2f}')
                                        print(f'lambda2         : {params["lambda2"]:.2f}')
                                        print(f'L1 (starting)   : {params["L1"]:.2f}')
                                        print(f'L2 (starting)   : {params["L2"]:.2f}')
                                        print(f'tol iPiano      : {params["tol_ipiano"]:.2e}')
                                        print(f'maxiter         : {params["N"]:03}')
                                        print('-- Primal-Dual --')
                                        print(f'tol PD          : {params["tol_primal_dual"]:.2e}')
                                        print(f'maxiter PD      : {params["T"]:03}')

                                        with open(params['fileID'], 'w') as f:
                                            f.write('\n########################\n')
                                            f.write('###### TV-OSMOSIS ######\n')
                                            f.write('########################\n\n')

                                            f.write(f'experiment: {experiment}\n')

                                            if params['flag_blur']:
                                                f.write(f'alpha blur: {BLUR[blur_test]}\n')
                                            else:
                                                f.write('alpha blur: no\n')

                                            if params['flag_initialisation'] == 0:
                                                f.write('initial.  : u0 = f\n')
                                            elif params['flag_initialisation'] == 1:
                                                f.write('initial.  : u0 = alpha*f + (1-alpha)*b\n')
                                            elif params['flag_initialisation'] == 2:
                                                f.write('initial.  : u0 = alpha*avg(f) + (1-alpha)*avg(b)\n')

                                            f.write('\n###### PARAMETERS ######\n')
                                            f.write('-- model --\n')
                                            f.write(f'eta             : {params["eta"]:.2f}\n')
                                            f.write(f'mu              : {params["mu"]:.2f}\n')
                                            f.write(f'gamma           : {params["gamma"]:.2f}\n')
                                            f.write(f'epsilon (huber) : {params["epsilon"]:.2f}\n')
                                            f.write('-- iPiano --\n')
                                            f.write(f'beta1           : {params["beta1"]:.2f}\n')
                                            f.write(f'beta2           : {params["beta2"]:.2f}\n')
                                            f.write(f'lambda1         : {params["lambda1"]:.2f}\n')
                                            f.write(f'lambda2         : {params["lambda2"]:.2f}\n')
                                            f.write(f'L1 (starting)   : {params["L1"]:.2f}\n')
                                            f.write(f'L2 (starting)   : {params["L2"]:.2f}\n')
                                            f.write(f'tol iPiano      : {params["tol_ipiano"]:.2e}\n')
                                            f.write(f'maxiter         : {params["N"]:03d}\n')
                                            f.write('-- Primal-Dual --\n')
                                            f.write(f'tol PD          : {params["tol_primal_dual"]:.2e}\n')
                                            f.write(f'maxiter PD      : {params["T"]:03d}\n')

                                        # add offset for positivity
                                        params['offset'] = 1
                                        foreground += params['offset']
                                        background += params['offset']

                                        # CORE ALGORITHM (each colour channel is processed separately)
                                        t_start = time.time()
                                        for c in range(C):
                                            params['c'] = c
                                            ufinal[:, :, c], vfinal[:, :, c], E[:, c] = TV_osmosis_wrapper(foreground[:, :, c], background[:, :, c], alpha[:, :, c], params)
                                        t_end = time.time()
                                        # t_end = time.time() - t_start

                                        # remove offset
                                        foreground -= params['offset']
                                        background -= params['offset']
                                        ufinal -= params['offset']
                                        vfinal -= params['offset']

                                        params_file = open(filename_txt, 'a')  # Open the file in append mode
                                        params_file.write('\ncputime: %f s.' % t_end)
                                        params_file.close()

                                        # Save images
                                        Image.fromarray((ufinal * 255).astype(np.uint8)).save(filename_u + '_time' + str(t_end) + '.png')
                                        Image.fromarray((vfinal * 255).astype(np.uint8)).save(filename_v + '_time' + str(t_end) + '.png')

                                        # Implement visualization if necessary
                                        if params['plot_figures']:
                                            plt.figure()
                                            plt.subplot(1, 3, 1)
                                            plt.imshow(foreground, vmin=0, vmax=1)
                                            plt.title('foreground')
                                            plt.subplot(1, 3, 2)
                                            plt.imshow(background, vmin=0, vmax=1)
                                            plt.title('background')
                                            plt.subplot(1, 3, 3)
                                            plt.imshow(ufinal, vmin=0, vmax=1)
                                            plt.title('result')
                                            # plt.show()
                                            # plt.savefig(filename_v+'_time' + str(t_end) + '.png')

                                            # Save results
                                        # np.savez(f"{filename_u}.npz", ufinal=ufinal, vfinal=vfinal)

# error
def GCM(z):
    # Calculate the Geometric Chromaticity Mean (GCM)
    return np.cbrt(z[0] * z[1] * z[2])

def err(u1, u2):
    # Calculate the pixel-wise chromaticity error between two given images u1 and u2
    err_value = np.abs((u1 / GCM(u1)) - (u2 / GCM(u2)))

    return np.sum(np.square(err_value))
#
error = err(foreground, ufinal)
print('{:.10f}'.format(error))
#
# print("foreground")
# print(foreground)
# print("ufinal")
# print(ufinal)