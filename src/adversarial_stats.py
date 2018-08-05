import numpy as np
import visualization_utils

paths = [
    "point_clouds/succeeded_point_clouds_eps_2_0.npz"
]

files = visualization_utils.read_npz_files(paths)

for i, file in enumerate(files):
    print("File %d" % (i + 1))
    print("%d objects total" % file["x_original"].shape[0])
    print("%d points per object" % file["x_original"].shape[1])

    perturbed = ~np.isclose(file["x_original"], file["x_adv"])
    perturbed = np.any(perturbed, axis = 2)
    perturbed = np.sum(perturbed, axis = 1)
    print("%d points perturbed minimum" % np.min(perturbed))
    print("%d points perturbed maximum" % np.max(perturbed))
    print("%d points perturbed on average" % np.mean(perturbed))

    norm = np.linalg.norm(file["x_adv"], axis = 2)
    print("Minimum L2 norm: %.3f" % np.min(norm))
    print("Maximum L2 norm: %.3f" % np.max(norm))
    print("Average L2 norm: %.3f" % np.mean(norm))