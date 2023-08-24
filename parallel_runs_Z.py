import numpy as np
import scipy
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import time
import multiprocessing
import os
from copy import deepcopy
import numpy.ma as ma

from methodsMemoryDeposition import *
from main import main as memoryDeposition

def main():
    BigFolder = "SimResults"
    params = {
    "init_cond":       "single",
    "height":            np.inf,
    "dom":                20000,
    "ndim":                   1,
    "t_max":               1000,
    "r_0":                    1,
    "tau":                    1,
    "dt_snapshot":            1,          
    "n_ptcl_snapshot":    np.inf,
    "keep_all":           False, 
    "foldername":       BigFolder,
    "filename":        "result",
    "seed":                  1,          
    }

    try:
        os.chdir(params["foldername"])
    except FileNotFoundError:
        os.mkdir(params["foldername"])
        os.chdir(params["foldername"])

    params_list = []
    for r_0 in [0.1, 0.5, 0.7, 0.9, 0.99, 0.999, 0.9999, 1, 2, 4, 8, 16, 32]:
        for L in [200, 400, 800, 1600, 3200]:
                for seed in range(32):

                    params["r_0"] = r_0
                    params["L"] = L
                    params["seed"] = seed
                    params["foldername"] = 'SimResults/sim_%.2er0_%dL_%dseed'%(params['r_0'], params['dom'],int(params['seed']))
                    params_list.append(deepcopy(params))

    num_cores = multiprocessing.cpu_count()  # Get the number of available CPU cores

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(memoryDeposition, [params for params in params_list])
    return 1

if __name__ == "__main__":
    main()