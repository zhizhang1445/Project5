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
from mainMemoryDeposition import main as memoryDeposition

def main():
    params = {
    "height":               400,
    "dom":                10000,
    "ndim":                   1,
    "t_max":             100000,
    "r_0":                    1,
    "tau":                  0.5,
    "dt_snapshot":        np.inf,       
    "n_ptcl_snapshot":    np.inf,
    "foldername":  "SimResults",
    "filename":        "result",
    }

    mean_exit_times = []
    std_exit_times = []
    taus = []

    for i in np.arange(-1.5, 0.03, 0.01):
        tau = (10**i)
        params["tau"] = tau
        taus.append(tau)

        local_exit_time = []
        num_cores = multiprocessing.cpu_count()  # Get the number of available CPU cores

        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(memoryDeposition, [params for _ in range(num_cores)])

        for res in results:
            _ , times = res
            if len(times) > 1:
                local_exit_time.append(times[-1])
            else:
                local_exit_time.append(0)

        print(f"{tau} Done| Mean Exit Time {np.mean(local_exit_time)}")
        mean_exit_times.append(np.mean(local_exit_time)) 
        std_exit_times.append(np.std(local_exit_time))
    return mean_exit_times, std_exit_times, taus

if __name__ == "__main__":
    mean_exit_times, std_exit_times, taus = main()
    y_low = np.array(mean_exit_times) - np.array(std_exit_times)
    y_high = np.array(mean_exit_times) + np.array(std_exit_times)
    np.save("mean_exit_times.npy", mean_exit_times)
    np.save("std_exit_times.npy", std_exit_times)
    np.save("taus.npy", taus)

    plt.plot(taus, mean_exit_times, label='Data')
    plt.fill_between(taus, y_low, y_high, alpha=0.2, label='Error Area')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"$\tau$ [1/s]")
    plt.ylabel("Survival Time [s]")
    plt.title(r"Survival time for $r_0 = 1$, $L = 10^4$")
    plt.savefig("Survival Probability.png")