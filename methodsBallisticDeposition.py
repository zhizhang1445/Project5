import numpy as np
import scipy
import scipy.sparse as sparse
import matplotlib.pyplot as plt

#this is the main module with all the methods we need, updatesteps go there
def get_nearest_neighbors(matrix, index, self = False):
    """
    Returns the indexes of the nearest neighbors to a given index in a 2D matrix.

    Args:
        matrix (numpy.ndarray): The 2D matrix.
        index (tuple): The given index (row, column).

    Returns:
        list: The indexes of the nearest neighbors.
    """
    # Get the number of rows and columns in the matrix
    rows, cols = matrix.shape

    # Unpack the given index
    row, col = index

    # Define the offsets for the neighbors (up, down, left, right)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    if self:
        offset = offset.append((0, 0))

    # List to store the valid neighbor indexes
    neighbor_indexes = []

    # Check each offset to determine the neighbor indexes
    for offset in offsets:
        new_row = row + offset[0]
        new_col = col + offset[1]

        # Check if the new index is within the matrix bounds
        if 0 <= new_row < rows:
            new_row
        else: new_row = -1
        if 0 <= new_col < cols:
            new_col
        else: new_col = -1
            
        neighbor_indexes.append((new_row, new_col))

    return neighbor_indexes

def update_propensities(propensities, index, params, dt = 1):
    r = params["r"]
    propensities = np.power(r, dt)*propensities
    x_ind, y_ind = index

    propensities[x_ind, y_ind] = 1

    near_neighbors = get_nearest_neighbors(propensities, (x_ind, y_ind))
    for neighbor in near_neighbors:
        propensities[neighbor[0], neighbor[1]] = 1
    return propensities

def get_nonzero_propensities(propensities):
    index_flat = list(zip(*np.nonzero(propensities))) #list of all indexes to use for choice

    x_ind, y_ind = np.nonzero(propensities)
    nonzero_propensities = propensities[x_ind, y_ind]
    probability = nonzero_propensities/np.sum(nonzero_propensities)

    return index_flat, probability

def choose_from_tau(taus):
    x_ind, y_ind = np.unravel_index(np.argmin(taus), taus.shape)
    tau = taus[x_ind, y_ind]
    return (x_ind, y_ind), tau

def add_point(index, space, max_height):
    x_ind, y_ind = index
    near_neighbors = get_nearest_neighbors(max_height, (x_ind, y_ind))
    height_surrondings = [max_height[neighbor[0], neighbor[1]] for neighbor in near_neighbors]

    highest_pos = max([max(height_surrondings), max_height[x_ind, y_ind]+1])
    space[x_ind, y_ind, highest_pos] = 1
    max_height[x_ind, y_ind] = highest_pos
    return space, max_height

def plot_surface(surface, show = True, title = "Ballistic Deposition", 
                 colorbar = False, save = False, name = None):
    """
    Plots the surface with particles.
    
    Args:
        surface (np.array): A 2D array representing the surface/grid.
        save: A boolean to turn the saving feature on/off
    """
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