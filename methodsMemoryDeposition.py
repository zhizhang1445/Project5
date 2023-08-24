import numpy as np
import matplotlib.pyplot as plt
import json
from copy import deepcopy

#this is the main module with all the methods we need, updatesteps go there
def write2json(foldername, params):
    with open(foldername + '/params.json', 'w') as fp:
        json.dump(params, fp)

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

def plot_surface(surface, max_height=None, show = True, title = "Ballistic Deposition", 
                 colorbar = False, save = False, name = None):
    plt.figure()

    if (max_height is not None):
        plt.plot(max_height)
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




