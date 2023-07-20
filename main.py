import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
import imageio
import anndata as ad

#This is the main file to run scripts, please only put finished 
#code here so we don't create merge conflicts
from methodsMemoryDeposition import *

def main(params):
    width = params["dom"]
    d = params["ndim"]
    params["max_CDF"] = max_CDF(params)
    t = n_ptcls = n_snapshot = 0 
    max_height_time = []

    shape = tuple(width for _ in range(d))
    max_height_flat = np.zeros((np.power(width, d)), dtype=int) #occupation/height at each site
    t_next = np.array([single_time(0, params) for _ in range(np.power(width, d))])

    while(t < params["t_max"]):
        try:
            index_chosen = np.argmin(t_next)
            t_min = t_next[index_chosen]
            max_height_flat = add_point_ndarray(index_chosen, max_height_flat, shape)
            
            if t_min == np.inf:
                print(f"EVERYONE IS DEAD AT: {t} | N_Ptcls: {n_ptcls}| N_snapshots: {n_snapshot}")
                break
            
            neighbors = get_nearest_non_diagonal_neighbors(index_chosen, shape)
            for index_ngbh in neighbors:
                t_next[index_ngbh] = single_time(t_min, params)
            t_next[index_chosen] = single_time

            t = t_min

            if (
                t > n_snapshot*params["dt_snapshot"]
                ) or (
                    n_ptcls%params["n_ptcl_snapshot"] == 0
                    ):
                max_height_time.append(deepcopy(max_height_flat))
                
                n_snapshot += 1
            n_ptcls += 1

        except KeyboardInterrupt:
            print(f"Manually Stopped at time: {t}| N_Ptcls: {n_ptcls}| N_snapshots: {n_snapshot}")
            break
    else:
        print(f"Stopped at time: {t}| N_Ptcls: {n_ptcls}| N_snapshots: {n_snapshot}")
    return max_height_time

if __name__ == "__main__":
    params = {
    "height":               400,
    "dom":                  400,
    "ndim":                   1,
    "t_max":                100,
    "r_0":                 0.01,
    "tau":                    1,
    "dt_snapshot":            1,          
    "n_ptcl_snapshot":       10,
    "foldername":  "SimResults",
    "filename":        "result2",
    }

    max_height_time = main1D_w_plotting(params)
    # adata = ad.AnnData(np.array(max_height_time).squeeze())

    # adata.uns.update(params)
    # foldername = params["foldername"]
    # filename = params["filename"]
    # adata.write(f"./{foldername}/{filename}.h5ad")