import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
import anndata as ad

#This is the main file to run scripts, please only put finished 
#code here so we don't create merge conflicts
from methodsMemoryDeposition import *

def main(params):
    if params['seed'] is None:
        np.random.seed(); params['seed'] = 0
    else:
        np.random.seed(params['seed'])
        
    width = params["dom"]
    d = params["ndim"]
    params["max_CDF"] = max_CDF(params)
    t = n_ptcls = n_snapshot = 0
    outfile = open('sim_%.2er0_%dL_%dseed.csv'%(params['r_0'], params['dom'],int(params['seed'])),'w')
    print('t,N,h_mean,h_std',file=outfile)

    max_height_time = []
    times = []

    shape = tuple(width for _ in range(d))
    max_height_flat = np.zeros((np.power(width, d)), dtype=int) #occupation/height at each site
    
    if params["init_cond"] == "homogenous":
        t_next = np.array([single_time(0, params) for _ in range(np.power(width, d))])
    elif params["init_cond"] == "single":
        t_next = np.full(np.power(width, d), np.inf)

        while(t_next[int(len(t_next)/2)] == np.inf):
            t_next[int(len(t_next)/2)] = single_time(0, params)
    else:
        raise NameError("Initialization type is wrong")

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
            t_next[index_chosen] = single_time(t_min, params)

            t = t_min

            if (
                t > n_snapshot*params["dt_snapshot"]
                ) or (
                    n_ptcls%params["n_ptcl_snapshot"] == 0
                    ):
                
                if params["keep_all"]:
                    max_height_time.append(deepcopy(max_height_flat))
                    times.append(t)

                print(t,n_ptcls,max_height_flat.mean(), max_height_flat.std(),sep=',',file=outfile)

                n_snapshot += 1
            n_ptcls += 1

        except KeyboardInterrupt:
            print(f"Manually Stopped at time: {t}| N_Ptcls: {n_ptcls}| N_snapshots: {n_snapshot}")
            break
    else:
        print(f"Stopped at time: {t}| N_Ptcls: {n_ptcls}| N_snapshots: {n_snapshot}")
    return max_height_time, times

if __name__ == "__main__":
    params = {
    "init_cond":      "single",
    "height":               100,
    "dom":                  100,
    "ndim":                   1,
    "t_max":                 10,
    "r_0":                    4,
    "tau":                    1,
    "dt_snapshot":            1,          
    "n_ptcl_snapshot":    np.inf,
    "keep_all":            False, 
    "foldername":   "SimResults",
    "filename":     "TestSingle",
    "seed":                None,
    }

    max_height_time, times = main1D_w_plotting(params)
