import numpy as np
import matplotlib.pyplot as plt
import json
import random
from copy import deepcopy

#this is the main module with all the methods we need, updatesteps go there
def write2json(foldername, params):
    with open(foldername + '/params.json', 'w') as fp:
        json.dump(params, fp)

def calc_empty_clusters(space_flat, max_height_flat, params, num_samples = np.inf):
    def dfs(start_index_double_flat): #Use Depth-First Search for entire fucking lattice
        empty_space_flag = False

        stack = [start_index_double_flat] #Using a stack to do DFS (dynamic)
        current_cluster = [start_index_double_flat] #Current cluster stack (accumulating)
        # print(len(space_double_flat_left))

        while stack:
            current_index = stack.pop()

            for neighbor in get_NNDN_and_time(current_index, params):
                if neighbor in space_double_flat_left: #space_double_flat_left checks if visited too
                    stack.append(neighbor)
                    space_double_flat_left.remove(neighbor)
                    # print(len(space_double_flat_left))
                    current_cluster.append(neighbor)

                if neighbor in known_empty_space: #Remove shit if it's neighbors is outside max height
                    empty_space_flag = True
                    
        return current_cluster, empty_space_flag
    
    def get_nonzero_below_MaxHeight(space_flat, max_height_flat): #This is to get all nonzero elements and things above max height
        space_double_flat_left = []
        known_empty_space = []
        shape_prev_flat = (np.power(params['dom'], params["ndim"]), params["height"])
        for x, (space_slice, max_h_for_slice) in enumerate(zip(space_flat, max_height_flat)): #I genuinely don't know if there's anything faster than this as this takes ~n*T
            for h, pt in enumerate(space_slice):
                if pt == 0:
                    flat_index = np.ravel_multi_index([x, h], shape_prev_flat)

                    if h <= max_h_for_slice:
                        space_double_flat_left.append(flat_index)
                    elif h> max_h_for_slice :
                        known_empty_space.append(flat_index)
        return space_double_flat_left, known_empty_space

    def get_NNDN_and_time(current_index_double_flat, params): #Returns double flat index of nearest non diagonal neighbors and also return itself
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
            
    space_double_flat_left, known_empty_space = get_nonzero_below_MaxHeight(space_flat, max_height_flat) # find all double flat indexes in space time
    list_empty_clusters = []

    while space_double_flat_left:
        rand_idx = random.randrange(len(space_double_flat_left)) #Taking a random index to start is actually much better for random small samples
        start_idx = space_double_flat_left.pop(rand_idx)
        
        current_cluster, empty_space_flag = dfs(start_idx)
        if not empty_space_flag:
            list_empty_clusters.append(current_cluster)
        
        if len(list_empty_clusters) >= num_samples: #You can decide to just sample 4 clusters and call it a day lmao
            print(f"Reached {num_samples} clusters, now stopping for conserving compute")
            break

    return list_empty_clusters

def unflat_empty_clusters(list_empty_clusters, params): # this function is to unflatten the double flatten space but also to remove clusters with known space
    list_cluster_single_flat = []
    shape_prev_flat = (np.power(params["dom"], params["ndim"]), params["height"])

    for cluster in list_empty_clusters:
        simple_flag = False

        if params["init_cond"] == "single":
            for i in range(params["ndim"]):
                idx = np.ravel_multi_index((i*params["dom"], 0), shape_prev_flat)
                if idx in cluster:
                    simple_flag = True
                    break
        
        if simple_flag:
            continue
        
        cluster_single_flat = []
        for coord_double_flat in cluster:
            coord_single_flat = np.unravel_index(coord_double_flat, shape_prev_flat)

            if coord_single_flat[1] == params["height"]-1: # Everything connected to last layer is removed
                simple_flag = True
                break #break out of the loop over coordinates
            cluster_single_flat.append(coord_single_flat)
    
        if simple_flag: #Skips the appending, removes a cluster if flag is trigerred
            continue
        list_cluster_single_flat.append(cluster_single_flat)
    return list_cluster_single_flat

def calc_MVS_empty_clusters(list_clusters_double_flat, params):
    if len(list_clusters_double_flat) == 0:
        return 0, 0, 0
    shape_prev_flat = (np.power(params["dom"], params["ndim"]), params["height"])
    Masses = []
    Volumes = []
    Sizes = []

    for cluster in list_clusters_double_flat:
        simple_flag = False
        if 0 in cluster and (params["init_cond"] == "single"):
            continue 

        cluster_single_flat = []
        cluster_times = []
        cluster_volumes = np.zeros(params["height"])

        for coord_double_flat in cluster:
            coord_single_flat = np.unravel_index(coord_double_flat, shape_prev_flat)
            height_index = coord_single_flat[1]

            if coord_single_flat[1] == params["height"]-1: # Everything connected to last layer is removed
                simple_flag = True
                break #break out of the loop over coordinates

            cluster_volumes[height_index] += 1
            cluster_times.append(height_index)
        
        if simple_flag: #Skips the appending, removes a cluster if flag is trigerred
            continue

        Masses.append(len(cluster)) #These do not happen if the flag is on. 
        Volumes.append(np.mean(cluster_volumes[np.nonzero(cluster_volumes)]))
        Sizes.append(np.max(cluster_times)-np.min(cluster_times))
    try:
        mass = np.mean(Masses)
        volume = np.mean(Volumes)
        size = np.mean(Sizes)
    except RuntimeWarning:
        mass = 0
        volume = 0
        size = 0
    return mass, volume, size

def calc_corr_length(max_height, params):
    def calc_trans_len_1D(array):
        start = end = 0
        for i,val in enumerate(array):
            if val > 0:
                if start == 0:
                    start = i

                end = i

        trans_length = end-start+1
        return trans_length
    
    ndim = params["ndim"]

    if ndim > 1:
        trans_lengths_by_axis = []

        for axis in range(ndim):
            slice_max_height = max_height[axis*params["dom"]:(axis+1)*params["dom"]]
            trans_lengths_by_axis.append(calc_trans_len_1D(slice_max_height))

        transverse_length = max(trans_lengths_by_axis)

    else:
        transverse_length = calc_trans_len_1D(max_height)

    parallel_length = np.max(max_height)

    if parallel_length <= 0:
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
        if params["time_dist_type"] == "continuous":
            time = quantile_Function(random_number, t_last, params)
        elif params["time_dist_type"] == "discrete":
            time = t_last +1
        else:
            raise KeyError("time_dist_type key does not work")
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


