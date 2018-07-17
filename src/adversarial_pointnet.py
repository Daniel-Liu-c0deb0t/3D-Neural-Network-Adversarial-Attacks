import numpy as np
import tensorflow as tf
import scipy
import adversarial_utils
import os
import sys
import importlib
working_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(working_dir, "models"))
sys.path.append(os.path.join(working_dir, "utils"))
import provider
import pc_util

model = importlib.import_module("pointnet_cls")
class_names = [line.rstrip() for line in open("data/modelnet40_ply_hdf5_2048/shape_names.txt")]
test_files = provider.getDataFiles("data/modelnet40_ply_hdf5_2048/test_files.txt")

num_points = 1024

data_x = []
data_t = []
for file in test_files:
    curr_x, curr_t = provider.loadDataFile(file)
    data_x.append(curr_x[:, :num_points, :])
    data_t.append(np.squeeze(curr_t))

data_x = np.concatenate(data_x)
data_t = np.concatenate(data_t)

x_pl, t_pl = model.placeholder_inputs(1, num_points)

is_training = tf.placeholder(tf.bool, shape = [])

def model_loss_fn(x, t):
    with tf.variable_scope("model", reuse = tf.AUTO_REUSE):
        y, end_points = model.get_model(x, is_training)
    if t is None:
        loss = None
    else:
        loss = model.get_loss(y, t, end_points)
    return y, loss

targeted = False
iter = 10
eps_list = [0.05, 0.1]
model_path = "log/model.ckpt"

if targeted:
    output_path = "adversary/targeted"
    x_original, target, x_adv = adversarial_utils.targeted_attack(model_path, output_path, x_pl, t_pl, model_loss_fn, data_x, data_t, class_names, iter = iter, eps_list = eps_list, one_hot = False, extra_feed_dict = {is_training: False})

    for eps_idx in range(len(eps_list)):
        class_idx = np.random.choice(len(class_names), size = 10, replace = False)
        for i in class_idx:
            idx = np.random.choice(len(x_original[eps_idx][i]), size = 3, replace = False)
            for j in idx:
                img_file = "%d_%s_original.jpg" % (j, class_names[target[eps_idx][i][j]])
                img_file = os.path.join(output_path, img_file)
                img = pc_util.point_cloud_three_views(x_original[eps_idx][i][j])
                scipy.misc.imsave(img_file, img)

                eps_str = str(eps_list[eps_idx]).replace(".", "_")
                img_file = "%d_%s_adv_target_%s_eps_%s.jpg" % (j, class_names[target[eps_idx][i][j]], class_names[i], eps_str)
                img_file = os.path.join(output_path, img_file)
                img = pc_util.point_cloud_three_views(x_adv[eps_idx][i][j])
                scipy.misc.imsave(img_file, img)
else:
    output_path = "adversary/untargeted"
    x_original, target, x_adv, pred_adv = adversarial_utils.untargeted_attack(model_path, output_path, x_pl, t_pl, model_loss_fn, data_x, data_t, class_names, iter = iter, eps_list = eps_list, one_hot = False, extra_feed_dict = {is_training: False})

    for eps_idx in range(len(eps_list)):
        idx = np.random.choice(len(x_original[eps_idx]), size = 10, replace = False)
        for i in idx:
            img_file = "%d_%s_original.jpg" % (i, class_names[target[eps_idx][i]])
            img_file = os.path.join(output_path, img_file)
            img = pc_util.point_cloud_three_views(x_original[eps_idx][i])
            scipy.misc.imsave(img_file, img)

            eps_str = str(eps_list[eps_idx]).replace(".", "_")
            img_file = "%d_%s_adv_pred_%s_eps_%s.jpg" % (i, class_names[target[eps_idx][i]], class_names[pred_adv[eps_idx][i]], eps_str)
            img_file = os.path.join(output_path, img_file)
            img = pc_util.point_cloud_three_views(x_adv[eps_idx][i])
            scipy.misc.imsave(img_file, img)