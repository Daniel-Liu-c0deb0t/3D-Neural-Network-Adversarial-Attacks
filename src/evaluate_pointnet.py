import numpy as np
import tensorflow as tf
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

numpy_file = True
num_points = 1024
model_path = "log/model.ckpt"
output_path = "evaluate"

if numpy_file:
    file = np.load("point_clouds.npz")
    data_x = file["points"][:, :num_points, :]
    data_t = file["labels"]
else:
    test_files = provider.getDataFiles("data/modelnet40_ply_hdf5_2048/test_files.txt")

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

adversarial_utils.evaluate(model_path, output_path, x_pl, t_pl, model_loss_fn, data_x, data_t, class_names, one_hot = False, extra_feed_dict = {is_training: False})