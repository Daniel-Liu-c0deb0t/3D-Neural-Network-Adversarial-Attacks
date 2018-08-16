import numpy as np
import tensorflow as tf
import scipy
import adversarial_utils
import os
import sys
import argparse
import importlib
working_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(working_dir, "models"))
sys.path.append(os.path.join(working_dir, "utils"))
import provider
import pc_util

parser = argparse.ArgumentParser(description = "Adversarial attacks on PointNet++ used for classification.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--checkpoint", default = "log/model.ckpt", help = "Path to the model's checkpoint file.")
parser.add_argument("--output", default = "adversarial", help = "Output directory.")
parser.add_argument("--data", default = "data/modelnet40_ply_hdf5_2048/test_files.txt", help = "Input data. Either a Numpy file or a text file containing a list of HDF5 files.")
parser.add_argument("--class-names", default = "data/modelnet40_ply_hdf5_2048/shape_names.txt", help = "Text file containing a list of class names.")
parser.add_argument("--num-points", type = int, default = 1024, help = "Number of points to use.")
parser.add_argument("--num-objects", type = int, default = 1000000000, help = "Number of correctly classified objects to use. Specify a very large number to use all correctly classified objects.")
parser.add_argument("--targeted", action = "store_true", help = "Run targeted attack.")
parser.add_argument("--iter", type = int, default = 10, help = "Number of iterations for iterative gradient sign.")
parser.add_argument("--eps", nargs = "+", type = float, default = [1], help = "List of epsilon values for iterative gradient sign.")
parser.add_argument("--mode", choices = ["iterative", "momentum", "saliency", "sort"], default = "iterative", help = "Which algorithm to use when perturbing points.")
parser.add_argument("--projection", action = "store_true", help = "Project the gradient vectors onto each point's corresponding triangle.")
parser.add_argument("--restrict", action = "store_true", help = "Restrict the gradient vectors to be inside each point's corresponding triangle.")
parser.add_argument("--norm", default = "inf", help = "Norm used for gradient sign.")
parser.add_argument("--clip-norm", type = float, default = None, help = "Value to clip L2 norm by.")
parser.add_argument("--min-norm", type = float, default = 0.0, help = "Ignore perturbations with a smaller L2 norm than this.")
args = parser.parse_args()
print(args)

model = importlib.import_module("pointnet2_cls_ssg")
class_names = [line.rstrip() for line in open(args.class_names)]

np.random.seed(0) # fixed seed for consistency

numpy_file = args.data.endswith(".npz")

if numpy_file:
    with np.load(args.data) as file:
        data_x = file["points"][:, :args.num_points, :]
        if args.projection:
            data_f = file["faces"][:, :args.num_points, :3, :]
        else:
            data_f = None
        data_t = file["labels"]
else:
    test_files = provider.getDataFiles(args.data)

    data_x = []
    data_t = []
    for file in test_files:
        curr_x, curr_t = provider.loadDataFile(file)
        data_x.append(curr_x[:, :args.num_points, :])
        data_t.append(np.squeeze(curr_t))

    data_x = np.concatenate(data_x)
    data_f = None
    data_t = np.concatenate(data_t)

x_pl, t_pl = model.placeholder_inputs(1, args.num_points)

is_training = tf.placeholder(tf.bool, shape = [])

def model_loss_fn(x, t):
    with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
        y, end_points = model.get_model(x, is_training)
    if t is None:
        loss = None
    else:
        loss = model.get_loss(y, t, end_points)
    return y, loss

if args.targeted:
    res = adversarial_utils.targeted_attack(args.checkpoint, args.output, x_pl, t_pl, model_loss_fn, data_x, data_t, args.num_objects, class_names, data_f = data_f, restrict = args.restrict, iter = args.iter, eps_list = args.eps, norm = args.norm, mode = args.mode, one_hot = False, clip_norm = args.clip_norm, min_norm = args.min_norm, extra_feed_dict = {is_training: False})
    if data_f is None:
        x_original, target, x_adv = res
    else:
        x_original, target, x_adv, faces = res

    for eps_idx in range(len(args.eps)):
        class_idx = np.random.choice(len(class_names), size = len(class_names), replace = False)
        for i in class_idx:
            idx = np.random.choice(len(x_original[eps_idx][i]), size = min(3, len(x_original[eps_idx][i])), replace = False)
            for j in idx:
                img_file = "%d_%s_original.jpg" % (j, class_names[target[eps_idx][i][j]])
                img_file = os.path.join(args.output, img_file)
                img = pc_util.point_cloud_three_views(x_original[eps_idx][i][j])
                scipy.misc.imsave(img_file, img)

                eps_str = str(args.eps[eps_idx]).replace(".", "_")
                img_file = "%d_%s_adv_target_%s_eps_%s.jpg" % (j, class_names[target[eps_idx][i][j]], class_names[i], eps_str)
                img_file = os.path.join(args.output, img_file)
                img = pc_util.point_cloud_three_views(x_adv[eps_idx][i][j])
                scipy.misc.imsave(img_file, img)
else:
    res = adversarial_utils.untargeted_attack(args.checkpoint, args.output, x_pl, t_pl, model_loss_fn, data_x, data_t, args.num_objects, class_names, data_f = data_f, restrict = args.restrict, iter = args.iter, eps_list = args.eps, norm = args.norm, mode = args.mode, one_hot = False, clip_norm = args.clip_norm, min_norm = args.min_norm, extra_feed_dict = {is_training: False})
    if data_f is None:
        x_original, target, x_adv, pred_adv = res
    else:
        x_original, target, x_adv, pred_adv, faces = res

    for eps_idx in range(len(args.eps)):
        idx = np.random.choice(len(x_original[eps_idx]), size = min(30, len(x_original[eps_idx])), replace = False)
        for i in idx:
            img_file = "%d_%s_original.jpg" % (i, class_names[target[eps_idx][i]])
            img_file = os.path.join(args.output, img_file)
            img = pc_util.point_cloud_three_views(x_original[eps_idx][i])
            scipy.misc.imsave(img_file, img)

            eps_str = str(args.eps[eps_idx]).replace(".", "_")
            img_file = "%d_%s_adv_pred_%s_eps_%s.jpg" % (i, class_names[target[eps_idx][i]], class_names[pred_adv[eps_idx][i]], eps_str)
            img_file = os.path.join(args.output, img_file)
            img = pc_util.point_cloud_three_views(x_adv[eps_idx][i])
            scipy.misc.imsave(img_file, img)