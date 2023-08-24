import numpy as np
import os
import matplotlib.pyplot as plt
import imageio

#This is the main file to run scripts, please only put finished 
#code here so we don't create merge conflicts
from methodsMemoryDeposition import *

def main1D_w_plotting(params):
    height = params["height"]
    width = params["dom"]
    d = 1
    
    params["max_CDF"] = cutoff = max_CDF(params)
    rho = params["r_0"]*params["tau"]
    t = n_ptcls = n_snapshot = 0 
    max_height_time = []
    times = []

    foldername = params["foldername"]
    filename = params["filename"]

    shape = tuple(width for _ in range(d))
    max_height = np.zeros((width), dtype=int) #occupation/height at each site
    space = np.zeros((width, height), dtype= int)

    if params["init_cond"] == "homogenous":
        t_next = np.array([single_time(0, params) for _ in range(np.power(width, d))])
    elif params["init_cond"] == "single":
        t_next = np.full(np.power(width, d), np.inf)
        while(t_next[int(len(t_next)/2)] == np.inf):
            t_next[int(len(t_next)/2)] = single_time(0, params)

    while(t<params["t_max"]):
        try:
            index_chosen = np.argmin(t_next)
            t_min = t_next[index_chosen]

            if t_min == np.inf:
                print(f"EVERYONE IS DEAD AT: {t} | N_Ptcls: {n_ptcls}| N_snapshots: {n_snapshot}")
                break
            
            space, max_height = add_point(index_chosen, space, max_height)
            
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
                    max_height_time.append(max_height)
                    times.append(t)

                    plot_surface(space.transpose(), max_height,
                                title = r"$\rho = $" + f"{rho:.2f}  " +  r"$t = $" + f"{t:.0f}",
                                save = True,
                                show = False,
                                name = f"./{foldername}/frame_{n_snapshot}")
                    n_snapshot += 1
            n_ptcls += 1

        except IndexError:
            if (np.max(max_height) == params["height"]-1):
                print(f"Fully Occupied at time: {t}| N_Ptcls: {n_ptcls}| N_snapshots: {n_snapshot}")
            else:
                raise IndexError(f"This IndexError is NOT expected:{t}| N_Ptcls: {n_ptcls}| N_snapshots: {n_snapshot}")
            break

        except KeyboardInterrupt:
            print(f"Manually Stopped at time: {t}| N_Ptcls: {n_ptcls}| N_snapshots: {n_snapshot}")
            break
    else:
        print(f"Stopped at time: {t}| N_Ptcls: {n_ptcls}| N_snapshots: {n_snapshot}")

    frames = []
    n_updates = 0

    while(True):
        try:
            image = imageio.v2.imread(f'./{foldername}/frame_{n_updates}.png')
            os.remove(f'./{foldername}/frame_{n_updates}.png')
            frames.append(image)
            n_updates += 1
        except FileNotFoundError:
            imageio.mimsave(f'./{foldername}/{filename}.gif', 
                        frames, fps = 30)
            break
    return 1

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

    main1D_w_plotting(params)