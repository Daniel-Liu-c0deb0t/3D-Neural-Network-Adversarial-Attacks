import numpy as np
import visualization_utils
from collections import defaultdict

paths = [
    "point_clouds/pointnet/saliency_untargeted_iter_l2/succeeded_point_clouds_eps_1_0.npz"
]

files = visualization_utils.read_npz_files(paths)

for i, file in enumerate(files):
    print("File %d" % (i + 1))
    if "x_adv" in file:
        print("%d objects total" % file["x_original"].shape[0])
        print("%d points per object" % file["x_original"].shape[1])

        perturbed = ~np.isclose(file["x_original"], file["x_adv"])
        perturbed = np.any(perturbed, axis = 2)
        perturbed = np.sum(perturbed, axis = 1)
        print("%d points perturbed minimum" % np.min(perturbed))
        print("%d points perturbed maximum" % np.max(perturbed))
        print("%d points perturbed on average" % np.mean(perturbed))

        norm = np.linalg.norm(file["x_adv"], axis = 2)
        print("Min L2 norm: %.3f" % np.min(norm))
        print("Max L2 norm: %.3f" % np.max(norm))
        print("Avg L2 norm: %.3f" % np.mean(norm))

        same_pos = []
        for obj in file["x_adv"]:
            count = defaultdict(int)
            res = 0
            for point in obj:
                count[point.tobytes()] += 1
            for key in count:
                if count[key] > 1:
                    res += count[key] - 1
            same_pos.append(res)
        same_pos = np.array(same_pos)

        print("Min number of duplicate points: %.3f" % np.min(same_pos))
        print("Max number of duplicate points: %.3f" % np.max(same_pos))
        print("Avg number of duplicate points: %.3f" % np.mean(same_pos))
    elif "saliency" in file:
        print("%d objects total" % file["points"].shape[0])
        print("%d points per object" % file["points"].shape[1])

        dist = np.abs(np.max(file["saliency"], axis = 0) - np.min(file["saliency"], axis = 0))
        print("Avg number of equal saliency gradients: %d" % np.mean(np.sum(np.all(dist < 1e-4, axis = 2), axis = 1)))
        
        norm = np.linalg.norm(file["saliency"], axis = 3)
        dist = np.abs(np.max(norm, axis = 0) - np.min(norm, axis = 0))
        print("Avg number of equal saliency norms: %d" % np.mean(np.sum(dist < 1e-4, axis = 1)))
        print("Min saliency norm: %.3f" % np.min(norm))
        print("Max saliency norm: %.3f" % np.max(norm))
        print("Avg saliency norm: %.3f" % np.mean(norm))