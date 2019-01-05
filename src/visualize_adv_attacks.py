import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import visualization_utils
import os

class_str = "car"
camera = (0, 0)

names = [
    "fast_l2",
    "iter_l2",
    "iter_l2_norm",
    "iter_l2_proj",
    "fast_l2_normalized",
    "iter_l2_normalized"
]

pointnet_paths = [
    "point_clouds/pointnet/untargeted_16_final/untargeted_16_fast_l2/succeeded_point_clouds_eps_1_0.npz",
    #"point_clouds/pointnet/unique/untargeted_fast_sign/succeeded_point_clouds_eps_0_05.npz",
    "point_clouds/pointnet/untargeted_16_final/untargeted_16_iter_l2/succeeded_point_clouds_eps_1_0.npz",
    "point_clouds/pointnet/untargeted_16_final/untargeted_16_iter_l2_norm/succeeded_point_clouds_eps_1_0.npz",
    #"point_clouds/pointnet/unique/untargeted_iter_l2_min_norm/succeeded_point_clouds_eps_10_0.npz",
    "point_clouds/pointnet/untargeted_16_final/untargeted_16_iter_l2_proj/succeeded_point_clouds_eps_1_0.npz",
    "point_clouds/pointnet/untargeted_16_final/untargeted_16_fast_l2_normalized/succeeded_point_clouds_eps_0_05.npz",
    "point_clouds/pointnet/untargeted_16_final/untargeted_16_iter_l2_normalized/succeeded_point_clouds_eps_0_05.npz"
    #"point_clouds/pointnet/unique/untargeted_saliency/succeeded_point_clouds_eps_2_0.npz"
]

pointnet2_paths = [
    "point_clouds/pointnet2/untargeted_fast_l2/succeeded_point_clouds_eps_1_0.npz",
    "point_clouds/pointnet2/untargeted_fast_sign/succeeded_point_clouds_eps_0_05.npz",
    "point_clouds/pointnet2/untargeted_iter_l2/succeeded_point_clouds_eps_1_0.npz",
    "point_clouds/pointnet2/untargeted_iter_l2_clip_norm/succeeded_point_clouds_eps_5_0.npz",
    "point_clouds/pointnet2/untargeted_iter_l2_min_norm/succeeded_point_clouds_eps_5_0.npz",
    "point_clouds/pointnet2/untargeted_iter_l2_proj/succeeded_point_clouds_eps_5_0.npz",
    "point_clouds/pointnet2/untargeted_saliency/succeeded_point_clouds_eps_2_0.npz"
]

show_both = False
show_title = False
show_axis_numbers = False
class_names = [line.rstrip() for line in open("shape_names_unique.txt")]

pointnet_files = visualization_utils.read_npz_files(pointnet_paths)
if show_both:
    pointnet2_files = visualization_utils.read_npz_files(pointnet2_paths)
pointnet_data = visualization_utils.files_to_dicts(pointnet_files)
if show_both:
    pointnet2_data = visualization_utils.files_to_dicts(pointnet2_files)

common = visualization_utils.get_intersection(pointnet_data)
if show_both:
    common = common & visualization_utils.get_intersection(pointnet2_data)
common = list(common)
print("Number of objects in common: %d" % len(common))

pointnet_labels = visualization_utils.get_common_labels(pointnet_files, pointnet_data, common, class_names)
if show_both:
    pointnet2_labels = visualization_utils.get_common_labels(pointnet2_files, pointnet2_data, common, class_names)
    assert pointnet_labels == pointnet2_labels
print("Common object labels: %s" % pointnet_labels)

chosen_idx = pointnet_labels.index(class_str)
pointnet_object = visualization_utils.get_object(pointnet_files, pointnet_data, common[chosen_idx])
if show_both:
    pointnet2_object = visualization_utils.get_object(pointnet2_files, pointnet2_data, common[chosen_idx])
print("Chosen label: %s" % class_names[pointnet_object[0]["labels"]])

os.makedirs("point_clouds/images/%s" % class_str, exist_ok = True)

plt.figure(figsize = (12, 7))
plt.subplot(111, projection = "3d")
if show_title:
    plt.title("Original")
plt.gca().scatter(*pointnet_object[0]["x_original"].T, zdir = "y", c = "C0", s = 5)
plt.axis("scaled")
min = np.min(pointnet_object[0]["x_original"]) - 0.1
max = np.max(pointnet_object[0]["x_original"]) + 0.1
plt.gca().set_xlim(min, max)
plt.gca().set_ylim(min, max)
plt.gca().set_zlim(min, max)
plt.gca().view_init(*camera)
if not show_axis_numbers:
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.gca().set_zticklabels([])
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
plt.savefig("point_clouds/images/%s/original_%s.eps" % (class_str, class_str))
#plt.show()
plt.close()

if show_both:
    arr = zip(pointnet_object, pointnet2_object)
else:
    arr = pointnet_object
for i, p in enumerate(arr):
    if show_both:
        p1, p2 = p
    else:
        p1 = p
    plt.figure(figsize = (12, 7))

    if show_both:
        plt.subplot(121, projection = "3d")
    else:
        plt.subplot(111, projection = "3d")
    if show_title:
        plt.title("PointNet")

    close1 = np.all(np.isclose(p1["x_adv"], p1["x_original"]), axis = 1)
    c1 = np.repeat("C1", len(p1["x_adv"]))
    c1[close1] = "C0"

    plt.gca().scatter(*p1["x_adv"].T, zdir = "y", c = c1, s = 5)
    plt.axis("scaled")
    min = np.min(p1["x_adv"]) - 0.1
    max = np.max(p1["x_adv"]) + 0.1
    plt.gca().set_xlim(min, max)
    plt.gca().set_ylim(min, max)
    plt.gca().set_zlim(min, max)
    plt.gca().view_init(*camera)
    if not show_axis_numbers:
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.gca().set_zticklabels([])

    if show_both:
        plt.subplot(122, projection = "3d")
        if show_title:
            plt.title("PointNet++")

        close2 = np.all(np.isclose(p2["x_adv"], p2["x_original"]), axis = 1)
        c2 = np.repeat("C1", len(p2["x_adv"]))
        c2[close2] = "C0"

        plt.gca().scatter(*p2["x_adv"].T, zdir = "y", c = c2, s = 5)
        plt.axis("scaled")
        min = np.min(p2["x_adv"]) - 0.1
        max = np.max(p2["x_adv"]) + 0.1
        plt.gca().set_xlim(min, max)
        plt.gca().set_ylim(min, max)
        plt.gca().set_zlim(min, max)
        plt.gca().view_init(*camera)
        if not show_axis_numbers:
            plt.gca().set_xticklabels([])
            plt.gca().set_yticklabels([])
            plt.gca().set_zticklabels([])
    
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)

    plt.savefig("point_clouds/images/%s/adv_%s_%s_pred_%s.eps" % (class_str, class_str, names[i], class_names[p1["pred_adv"]]))
    #plt.show()
    plt.close()

    print("File %d" % (i + 1))
    print("PointNet adversarial prediction: %s" % class_names[p1["pred_adv"]])
    if show_both:
        print("PointNet++ adversarial prediction: %s" % class_names[p2["pred_adv"]])