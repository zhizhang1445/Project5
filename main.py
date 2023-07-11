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
    num_particles = params["num_particles"]
    r = params["r"]
    t = n_ptcls = n_updates = 0

    space = np.zeros((width, length, height), dtype=int) #actual simulation space
    max_height = np.zeros((width, length), dtype=int) #occupation/height at each site
    propensities = np.ones((width, length), dtype = float)

    adata = ad.AnnData()

    for key, value in params.items():
        setattr(adata, key, value)

    adata.write("./SimResults/result.h5ad")

    while(t<params["t_max"]):
        try:
            tau = 1/np.sum(propensities) * np.log(1/np.random.random())

            index_flat, probability = get_nonzero_propensities(propensities)
            choice = np.random.choice(len(index_flat), p = probability)
            x_ind, y_ind = index_flat[choice]

            near_neighbors = get_nearest_neighbors(propensities, index_flat[choice])
            # Find the highest position in the selected column
            height_surrondings = [max_height[neighbor[0], neighbor[1]] for neighbor in near_neighbors]

            highest_pos = max([max(height_surrondings), max_height[x_ind, y_ind]+1])

            # Deposit the particle at the next position above the highest position
            space[x_ind, y_ind, highest_pos] = 1
            max_height[x_ind, y_ind] = highest_pos
            
            t += tau

            if (
                t > n_updates*params["dt_update_time"]
                ) or (
                    num_particles%params["n_ptcl_update"] == 0
                    ):
                
                plot_surface(space[:, 0, :].transpose(),show = False, title=f"Ballistic Deposition time: {t}",  save= True, name = f"frame_{n_updates}.png")
                propensities = update_propensities(propensities, index_flat[choice], params)
                n_updates += 1
            n_ptcls += 1

        except KeyboardInterrupt or IndexError:
            print(f"Stopped at time: {t}| N_Ptcls: {n_ptcls}| N_updates: {n_updates}")
            break
    else:
        print(f"Stopped at time: {t}| N_Ptcls: {n_ptcls}| N_updates: {n_updates}")

    frames = []
    n_updates = 0
    foldername = "SimResults"

    while(True):
        try:
            image = imageio.v2.imread(f'./{foldername}/frame_{n_updates}.png')
            os.remove(f'./{foldername}/frame_{n_updates}.png')
            frames.append(image)
            n_updates += 1
        except FileNotFoundError: 
            break

    imageio.mimsave(f'./{foldername}/Animation_r_{params["r"]}.gif', 
                    frames, fps = 30)

if __name__ == "__main__":
    params = {
        "height":           400,
        "x_dom":            200,
        "y_dom":              1,
        "num_particles":   2000,
        "t_max":            100,
        "r":               0.75,
        "dt_update_time":     1,          
        "n_ptcl_update": np.inf,          
    }
    main(params)
    