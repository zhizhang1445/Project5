import numpy as np
import matplotlib.pyplot as plt
import json
from copy import deepcopy

#this is the main module with all the methods we need, updatesteps go there
def write2json(foldername, params):
    with open(foldername + '/params.json', 'w') as fp:
        json.dump(params, fp)

def calc_empty_zones(space_flat, params):
    def dfs(index): #Use Depth-First Search for entire fucking lattice
        stack = [index]
        
        visited.add(index)

        while stack:
            current_index = stack.pop()

            for neighbor in get_NNDN_and_time(current_index, params):
                if neighbor not in visited:
                    stack.append(neighbor)
                    visited.add(neighbor)
                    current_cluster.append(neighbor)

    def get_NNDN_and_time(current_index, params):
        shape = tuple(params["dom"] for _ in range(params["ndim"]))

        space_index_flat, time_index = current_index
        neighbors_same_layer = get_nearest_non_diagonal_neighbors(space_index_flat, shape)
        time_neighbor = [(space_index_flat, time_index)]

        for neighbor in neighbors_same_layer:
            if not time_index-1 < 0:
                time_neighbor.append((neighbor, time_index-1))
            
            time_neighbor.append((neighbor, time_index))

            if time_index < params["height"]:
                time_neighbor.append((neighbor, time_index+1))
            

    empty_indexes = np.where(space_flat==0)

    list_empty_clusters = []
    visited= set()

    for idx in empty_indexes:
        if idx not in visited and space_flat[idx] == 0:
            current_cluster = [idx]
            dfs(idx)
            list_empty_clusters.append(current_cluster)
        
    return list_empty_clusters

def calc_MVS_empty_clusters(list_empty_clusters, params):
    if params["init_cond"]:
        for cluster in list_empty_clusters:
            if (0,0) in cluster:
                list_empty_clusters.remove(cluster)
    return list_empty_clusters

def calc_corr_length(max_height, params):
    def calc_paral_len_1D(max_height):
        start = end = 0
        for i,val in enumerate(max_height):
            if val > 0:
                if start == 0:
                    start = i

                end = i

        parallel_length = end-start+1
        return parallel_length
    
    ndim = params["ndim"]

    if ndim > 1:
        parallel_lengths_by_axis = []

        for axis in range(ndim):
            slice_max_height = max_height[axis*params["dom"]:(axis+1)*params["dom"]]
            parallel_lengths_by_axis.append(calc_paral_len_1D(slice_max_height))

        parallel_length = max(parallel_lengths_by_axis)

    else:
        parallel_length = calc_paral_len_1D(max_height)

    if parallel_length == params["dom"]-1:
        parallel_length = np.NaN
    transverse_length = np.max(max_height)

    if transverse_length <= 0:
        raise IndexError("Something Went Wrong with the h_range")
    return transverse_length, parallel_length

def calculate_density(space_flat, global_max_height_prev, max_height_flat):
    global_max_height = np.max(max_height_flat) # Get height of last layer
    slice_space_flat = space_flat[global_max_height_prev:global_max_height+1] #Compute Density semi-crudely 
    #This should always include at least one layer for the space and not have a index problem because we check the height

    density = np.sum(slice_space_flat)/np.size(slice_space_flat)
    return density, global_max_height_prev

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
    return max_height_flat, highest_pos

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


