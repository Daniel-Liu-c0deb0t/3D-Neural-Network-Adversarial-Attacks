import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import visualization_utils

class_str = "tv_stand"

pointnet_paths = [
    "point_clouds/pointnet/untargeted_fast_l2/succeeded_point_clouds_eps_1_0.npz",
    "point_clouds/pointnet/untargeted_fast_sign/succeeded_point_clouds_eps_0_05.npz",
    "point_clouds/pointnet/untargeted_iter_l2/succeeded_point_clouds_eps_1_0.npz",
    "point_clouds/pointnet/untargeted_iter_l2_clip_norm/succeeded_point_clouds_eps_5_0.npz",
    "point_clouds/pointnet/untargeted_iter_l2_min_norm/succeeded_point_clouds_eps_5_0.npz",
    "point_clouds/pointnet/untargeted_iter_l2_proj/succeeded_point_clouds_eps_5_0.npz"
]

pointnet2_paths = [
    "point_clouds/pointnet2/untargeted_fast_l2/succeeded_point_clouds_eps_1_0.npz",
    "point_clouds/pointnet2/untargeted_fast_sign/succeeded_point_clouds_eps_0_05.npz",
    "point_clouds/pointnet2/untargeted_iter_l2/succeeded_point_clouds_eps_1_0.npz",
    "point_clouds/pointnet2/untargeted_iter_l2_clip_norm/succeeded_point_clouds_eps_5_0.npz",
    "point_clouds/pointnet2/untargeted_iter_l2_min_norm/succeeded_point_clouds_eps_5_0.npz",
    "point_clouds/pointnet2/untargeted_iter_l2_proj/succeeded_point_clouds_eps_5_0.npz"
]

class_names = [line.rstrip() for line in open("shape_names.txt")]

pointnet_files = visualization_utils.read_npz_files(pointnet_paths)
pointnet2_files = visualization_utils.read_npz_files(pointnet2_paths)
pointnet_data = visualization_utils.files_to_dicts(pointnet_files)
pointnet2_data = visualization_utils.files_to_dicts(pointnet2_files)

common = list(
    visualization_utils.get_intersection(pointnet_data) &
    visualization_utils.get_intersection(pointnet2_data)
)
print("Number of objects in common: %d" % len(common))

pointnet_labels = visualization_utils.get_common_labels(pointnet_files, pointnet_data, common, class_names)
pointnet2_labels = visualization_utils.get_common_labels(pointnet2_files, pointnet2_data, common, class_names)
assert pointnet_labels == pointnet2_labels
print("Common object labels: %s" % pointnet_labels)

chosen_idx = pointnet_labels.index(class_str)
pointnet_object = visualization_utils.get_object(pointnet_files, pointnet_data, common[chosen_idx])
pointnet2_object = visualization_utils.get_object(pointnet2_files, pointnet2_data, common[chosen_idx])
print("Chosen label: %s" % class_names[pointnet_object[0]["labels"]])

plt.figure(figsize = (12, 7))
plt.subplot(111, projection = "3d")
plt.title("Original")
plt.gca().scatter(*pointnet_object[0]["x_original"].T, zdir = "y", s = 5)
plt.axis("scaled")
plt.gca().set_xlim(-1, 1)
plt.gca().set_ylim(-1, 1)
plt.gca().set_zlim(-1, 1)
plt.gca().view_init(0, 0)
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
plt.savefig("point_clouds/images/%s/adversarial_image_original.png" % class_str)
#plt.show()
plt.close()

for i, (p1, p2) in enumerate(zip(pointnet_object, pointnet2_object)):
    plt.figure(figsize = (12, 7))

    plt.subplot(121, projection = "3d")
    plt.title("Pointnet")
    plt.gca().scatter(*p1["x_adv"].T, zdir = "y", s = 5)
    plt.axis("scaled")
    plt.gca().set_xlim(-1, 1)
    plt.gca().set_ylim(-1, 1)
    plt.gca().set_zlim(-1, 1)
    plt.gca().view_init(0, 0)

    plt.subplot(122, projection = "3d")
    plt.title("Pointnet++")
    plt.gca().scatter(*p2["x_adv"].T, zdir = "y", s = 5)
    plt.axis("scaled")
    plt.gca().set_xlim(-1, 1)
    plt.gca().set_ylim(-1, 1)
    plt.gca().set_zlim(-1, 1)
    plt.gca().view_init(0, 0)
    
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)

    plt.savefig("point_clouds/images/%s/adversarial_image_%d.png" % (class_str, (i + 1)))
    #plt.show()
    plt.close()

    print("File %d" % (i + 1))
    print("Pointnet adversarial prediction: %s" % class_names[p1["pred_adv"]])
    print("Pointnet++ adversarial prediction: %s" % class_names[p2["pred_adv"]])