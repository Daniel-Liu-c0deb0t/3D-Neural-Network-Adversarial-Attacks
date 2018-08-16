import numpy as np
import tensorflow as tf
import adversarial_utils
import os
import sys
import importlib
working_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(working_dir, "models"))
sys.path.append(os.path.join(working_dir, "utils"))

loss_saliency = True
saliency_class = 8
num_points = 1024
model_path = "log/model.ckpt"
out_dir = "saliency"
path = "point_clouds.npz"

model = importlib.import_module("pointnet_cls")
class_names = [line.rstrip() for line in open("data/modelnet40_ply_hdf5_2048/shape_names.txt")]

with np.load(path) as file:
    if "points" in file:
        data_x = file["points"][:, :num_points, :]
        data_t = file["labels"]
    else:
        data_x = file["x_adv"]
        data_t = file["labels"]

x_pl, t_pl = model.placeholder_inputs(1, num_points)

is_training = tf.placeholder(tf.bool, shape = [])

def model_loss_fn(x, t):
    with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
        y, end_points = model.get_model(x, is_training, num_classes = len(class_names))
    if t is None:
        loss = None
    else:
        loss = model.get_loss(y, t, end_points)
    return y, loss

if loss_saliency:
    saliency = adversarial_utils.saliency(model_path, x_pl, model_loss_fn, data_x, class_names, t_pl = t_pl, data_t = data_t, extra_feed_dict = {is_training: False})
else:
    saliency = adversarial_utils.saliency(model_path, x_pl, model_loss_fn, data_x, class_names, saliency_class = saliency_class, extra_feed_dict = {is_training: False})

np.savez_compressed(os.path.join(out_dir, "saliency.npz"), points = data_x, labels = data_t, saliency = saliency)