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
from main_Zhi import main as memoryDeposition

def main():
    params = { #Simulation Parameters
    "init_cond":       "single", #Set to "single" for single starting point percolation "homogenous" for full lattice start
    "height":              1000,
    "dom":                20000,
    "ndim":                   1,
    "t_max":              10000,
    "r_0":                    1,
    "tau":                    1,
    "dt_snapshot":           10,          
    "n_ptcl_snapshot":  np.Infinity,
    "keep_all":           False, 
    "foldername":     "../SimResults_Singles/",
    "filename":        "result",
    "seed":                  1, 
    "Whole_Lattice":      True,         
    }

    try:
        os.chdir(params["foldername"])
    except FileNotFoundError:
        os.mkdir(params["foldername"])
        os.chdir(params["foldername"])

    with open("runtime_stats.txt",'w') as outfile:
        print("Init_cond:", params["init_cond"],
            "|  height:", params["dom"],
            "|  ndim:", params["ndim"],
            "|  t max:", params["t_max"],
            "|  tau", params["tau"],
            file=outfile)

    params_list = []
    i = 1
    for r_0 in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 1, 1.001, 1.0001, 1.01, 1.05, 1.1, 1.3, 1.5, 1.7, 2]:
        for dom in [200, 400, 800, 1600]:
                init_random_seed = np.random.randint(0, 1000, 1)
                for seed in range(32):
                    temp_params = deepcopy(params)
                    temp_params["r_0"] = r_0
                    temp_params["dt_snapshot"] = 10*r_0
                    temp_params["dom"] = dom
                    temp_params["seed"] = int(seed*init_random_seed)
                    temp_params["foldername"] = params["foldername"]

                    if r_0 >= 1:
                        temp_params["t_max"] = 1000
                        temp_params["n_ptcl_snapshot"]= 1000
                    params_list.append(temp_params)
                    i+=1
                with open("runtime_stats.txt",'a') as outfile:
                    print("r0= ", r_0, " |  L= ", dom, "|   init seed used= ", init_random_seed, " to be done.", file=outfile)    

    num_cores = multiprocessing.cpu_count()-4  # Get the number of available CPU cores

    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(memoryDeposition, [params for params in params_list])
    return 1

if __name__ == "__main__":
    main()