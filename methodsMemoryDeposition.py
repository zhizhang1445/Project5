import numpy as np
import matplotlib.pyplot as plt
import json
from copy import deepcopy

#this is the main module with all the methods we need, updatesteps go there
def write2json(foldername, params):
    with open(foldername + '/params.json', 'w') as fp:
        json.dump(params, fp)

def calc_empty_zones_temp(space_flat, params):
    def dfs(start_index_double_flat, space_double_flat, visited): #Use Depth-First Search for entire fucking lattice
        stack = [start_index_double_flat]
        current_cluster = [start_idx]
        visited.append(start_index_double_flat)

        while stack:
            current_index = stack.pop()

            for neighbor in get_NNDN_and_time(current_index, params):
                if neighbor not in visited and (space_double_flat[neighbor]==0):
                    stack.append(neighbor)
                    visited.append(neighbor)
                    current_cluster.append(neighbor)
        return current_cluster, visited

    def get_NNDN_and_time(current_index_double_flat, params): #Returns double flat index and also return itself
        shape_prev_flat = (np.power(params['dom'], params["ndim"]), params["height"])
        shape_space = tuple(params["dom"] for _ in range(params["ndim"]))

        # print("Shape_prev_flat:", shape_prev_flat)
        space_index_flat, time_index = np.unravel_index(current_index_double_flat, shape_prev_flat)

        # print("Shape_space:", shape_space)
        neighbors_same_layer = get_nearest_non_diagonal_neighbors(space_index_flat, shape_space)

        double_flat_index = np.ravel_multi_index((space_index_flat, time_index), shape_prev_flat)
        time_neighbors = [double_flat_index]

        for neighbor in neighbors_same_layer:
            try:
                if not time_index-1 < 0:
                    double_flat_index = np.ravel_multi_index((neighbor, time_index-1), shape_prev_flat )
                    time_neighbors.append(double_flat_index)
                
                double_flat_index = np.ravel_multi_index((neighbor, time_index), shape_prev_flat )
                time_neighbors.append(double_flat_index)

                if not time_index+1 >= params["height"]: #This is so that the addition of one doesn't fuck me up
                    double_flat_index = np.ravel_multi_index((neighbor, time_index+1), shape_prev_flat)
                    time_neighbors.append(double_flat_index)
            except ValueError:
                print("DoubleFlatIndexError: |Space_index: ",neighbor,"TimeIndex: ", 
                      time_index, "CastShape: " ,shape_prev_flat)
        return time_neighbors
            
    space_double_flat = np.ravel(space_flat) #Double flat means flat in space and flat in time too
    empty_double_flat_indexes = np.argwhere(space_double_flat==0) # find all double flat indexes in space time

    list_empty_clusters = []
    visited= [] #global variable for tracking visited double flat indexes


    for start_idx in empty_double_flat_indexes:
        if start_idx not in visited:
            
            current_cluster, visited = dfs(start_idx, space_double_flat, visited)
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


