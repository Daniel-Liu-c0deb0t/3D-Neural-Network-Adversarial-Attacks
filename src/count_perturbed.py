import numpy as np
import visualization_utils

paths = [
    "point_clouds/succeeded_point_clouds_eps_2_0.npz"
]

files = visualization_utils.read_npz_files(paths)

for i, file in enumerate(files):
    perturbed = ~np.isclose(file["x_original"], file["x_adv"])
    perturbed = np.any(perturbed, axis = 2)
    perturbed = np.sum(perturbed, axis = 1)
    perturbed = np.mean(perturbed)
    print("File %d" % (i + 1))
    print("%d Points Perturbed!" % perturbed)