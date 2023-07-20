import numpy as np
import scipy
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import imageio
import os
from copy import deepcopy

#this is the main module with all the methods we need, updatesteps go there
def quantile_Function(random_numbers, t_last, params):
    r_0 = params["r_0"]
    tau = params["tau"]

    inside_part = (1/(r_0*tau))*np.log(1-random_numbers)
    return t_last - tau*np.log(1+inside_part)

def max_CDF(params):
    r_0 = params["r_0"]
    tau = params["tau"]
    return 1-np.exp(-1*tau*r_0)

def single_time(t_last, params):
    cutoff = params["max_CDF"]
    random_number = np.random.random()
    if random_number >= cutoff:
        return np.inf
    else:
        time = quantile_Function(random_number, t_last, params)
    return time

def get_nearest_non_diagonal_neighbors(index_flat, shape):
    neighbors = []
    output = []
    ndim = len(shape)

    # Extract indices from the input index
    if ndim == 1:
        for j in [-1, 1]:
            # Create a copy of the indices
            neighbor_indices = deepcopy(index_flat)
            neighbor_indices += j            
            if np.any(neighbor_indices < 0):
                neighbor_indices = shape[0]-1
            if np.any(neighbor_indices >= shape[0]):
                neighbor_indices = 0
            # Check if the neighbor indices are within bounds and not diagonal
            if not np.all(neighbor_indices == index_flat):
                neighbors.append(neighbor_indices)
        return neighbors
            
    else:
        indices = np.array(np.unravel_index(index_flat, shape))
        for i in range(ndim):
            for j in [-1, 1]:
                # Create a copy of the indices
                neighbor_indices = deepcopy(indices)
                neighbor_indices[i] += j

                if np.any(neighbor_indices[i] < 0):
                    neighbor_indices[i] = shape[j]-1
                if np.any(neighbor_indices[i] >= shape[j]):
                    neighbor_indices[i] = 0
                # Check if the neighbor indices are within bounds and not diagonal
                if not np.all(neighbor_indices == indices):
                    neighbors.append(tuple(neighbor_indices))

    for indices in neighbors:
        output.append(np.ravel_multi_index(indices, shape))
    return output

def add_point_ndarray(index, max_height_flat, shape:tuple = None):
    near_neighbors = get_nearest_non_diagonal_neighbors(index, shape)
    height_surrondings = [max_height_flat[neighbor] for neighbor in near_neighbors]

    highest_pos = max([max(height_surrondings), max_height_flat[index]+1])

    max_height_flat[index] = highest_pos
    return max_height_flat

def add_point(index, space, max_height):
    near_neighbors = get_nearest_non_diagonal_neighbors(index, max_height.shape)
    height_surrondings = [max_height[neighbor] for neighbor in near_neighbors]

    highest_pos = max([max(height_surrondings), max_height[index]+1])
    space[index, highest_pos] = 1
    max_height[index] = highest_pos
    return space, max_height

def plot_surface(surface, show = True, title = "Ballistic Deposition", 
                 colorbar = False, save = False, name = None):
    fig = plt.figure()
    plt.imshow(surface, cmap='binary', origin='lower')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    if colorbar:
        plt.colorbar()
    if save:
        if len(name) == 0:
           raise ValueError("No Name for surface plot")
        else: plt.savefig(name+".png")
        plt.close()
    # plt.show()
    if show:
        plt.show()

def main1D_w_plotting(params):
    height = params["height"]
    width = params["dom"]
    d = 1
    params["max_CDF"] = cutoff = max_CDF(params)
    t = n_ptcls = n_snapshot = 0 
    max_height_time = []
    foldername = params["foldername"]
    filename = params["filename"]

    shape = tuple(width for _ in range(d))
    max_height = np.zeros((width), dtype=int) #occupation/height at each site
    t_next = np.array([single_time(0, params) for _ in range(width)])
    space = np.zeros((width, height), dtype= int)

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

                    plot_surface(space.transpose(),
                                title = f"p: {cutoff} | time: {t}",  
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
    return max_height_time
