import numpy as np
import sys
sys.path.append('../')
from main_Zhi import main


r_0 = float(sys.argv[1])
L = int(sys.argv[2])
seed = int(sys.argv[3])

params = {
    "height":            np.inf,
    "dom":                    L,
    "ndim":                   1,
    "t_max":               1000,
    "r_0":                  r_0,
    "tau":                    1,
    "dt_snapshot":            1,          
    "n_ptcl_snapshot":   np.inf,
    "foldername":  "SimResults",
    "filename":        "result",
    "seed":                seed,
}

main(params)
