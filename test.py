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
from main_Tommaso import main as memoryDeposition

def main():
    params = {
    "height":               400,
    "dom":                10000,
    "ndim":                   1,
    "t_max":                100,
    "r_0":                    1,
    "tau":                  0.5,
    "dt_snapshot":        np.inf,       
    "n_ptcl_snapshot":      400,
    "foldername":  "SimResults",
    "filename":        "result",
    "seed":                  1,          
    }

    mean_exit_times = []
    std_exit_times = []
    taus = []

    for tau in [2, 4, 8, 16, 32]:
        for L in [200, 400, 800, 1600, 3200]:
                params_list = []
                for seed in range(32):

                    params["tau"] = tau
                    params["L"] = L
                    params["seed"] = seed
                    params_list.append(deepcopy(params))

                num_cores = multiprocessing.cpu_count()  # Get the number of available CPU cores

                with multiprocessing.Pool(processes=num_cores) as pool:
                    results = pool.map(memoryDeposition, [params for _ in range(num_cores)])
    return 1

if __name__ == "__main__":
    main()