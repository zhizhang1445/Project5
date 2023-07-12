import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
import imageio
import anndata as ad

#This is the main file to run scripts, please only put finished 
#code here so we don't create merge conflicts
from methodsBallisticDeposition import *

def main(params):
    height = params["height"]
    width = params["x_dom"]
    length = params["y_dom"]
    r = params["r"]
    foldername = params["foldername"]
    filename = params["filename"]
    t = n_ptcls = n_snapshot = 0 
    space_time = []
    propensities_time = []

    space = np.zeros((width, length, height), dtype=int) #actual simulation space
    max_height = np.zeros((width, length), dtype=int) #occupation/height at each site
    propensities = np.ones((width, length), dtype = float) #probability of droping at each coordinate
    taus = np.random.exponential(1/propensities) #purtiaty times generated from exponential distribution

    while(t<params["t_max"]):
        try:
            index_chosen, tau_min = choose_from_tau(taus)
            t += tau_min
            space, max_height = add_point(index_chosen, space, max_height)
            propensities_new = update_propensities(propensities, index_chosen, params, dt = tau_min)

            if params["all_resample"]:
                with np.errstate(divide="ignore"):
                    taus = np.random.exponential(1/propensities_new)
                    taus[np.isnan(taus)] = np.inf
            else:
                with np.errstate(all='ignore'):
                    taus = (np.divide(propensities, propensities_new)*(taus - tau_min))
                    taus[np.isnan(taus)] = np.inf

                taus[index_chosen[0], index_chosen[1]
                ] = np.random.exponential(1/propensities_new[index_chosen[0], 
                                                            index_chosen[1]])

            propensities = propensities_new
            
            if (
                t > n_snapshot*params["dt_snapshot"]
                ) or (
                    n_ptcls%params["n_ptcl_snapshot"] == 0
                    ):
                
                propensities_time.append(propensities)
                space_time.append(space)

                if params["gif"]:
                    surface = space[:, 0, :].transpose() if (params["y_dom"] == 1) else max_height

                    plot_surface(surface,
                                 title = f"r: {r} | time: {t}",  
                                 save = params["gif"],
                                 show = False,
                                 name = f"./{foldername}/frame_{n_snapshot}")
                n_snapshot += 1
            n_ptcls += 1

        except IndexError:
            if (np.max(max_height) == params["height"]-1):
                print(f"Fully Occupied at time: {t}| N_Ptcls: {n_ptcls}| N_snapshots: {n_snapshot}")
            else:
                raise IndexError("This IndexError is NOT expected")
            break

        except KeyboardInterrupt:
            print(f"Manually Stopped at time: {t}| N_Ptcls: {n_ptcls}| N_snapshots: {n_snapshot}")
            break
    else:
        print(f"Stopped at time: {t}| N_Ptcls: {n_ptcls}| N_snapshots: {n_snapshot}")

    frames = []
    n_updates = 0
    while(params["gif"]):
        try:
            image = imageio.v2.imread(f'./{foldername}/frame_{n_updates}.png')
            os.remove(f'./{foldername}/frame_{n_updates}.png')
            frames.append(image)
            n_updates += 1
        except FileNotFoundError:
            imageio.mimsave(f'./{foldername}/{filename}.gif', 
                        frames, fps = 30)
            break
    return space_time

if __name__ == "__main__":
    params = {
    "height":               400,
    "x_dom":                400,
    "y_dom":                  1,
    "t_max":                100,
    "r":                    0.3,
    "dt_snapshot":            1,          
    "n_ptcl_snapshot":   np.inf,
    "gif":                 True,
    "all_resample":       False,
    "foldername":  "SimResults",
    "filename":        "result2",
    }

    space_time = main(params)
    space_time = [space.flatten() for space in space_time if isinstance(space, np.ndarray) ]
    adata = ad.AnnData(np.array(space_time).squeeze())

    adata.uns.update(params)
    foldername = params["foldername"]
    filename = params["filename"]
    adata.write(f"./{foldername}/{filename}.h5ad")