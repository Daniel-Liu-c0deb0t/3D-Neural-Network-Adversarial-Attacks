import numpy as np
import tensorflow as tf
import adversarial_utils
import os
import sys
import argparse
import importlib
working_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(working_dir, "models"))
sys.path.append(os.path.join(working_dir, "utils"))
import provider

parser = argparse.ArgumentParser(description = "Gets the feature vector for Pointnet.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--checkpoint", default = "log/model.ckpt", help = "Path to the model's checkpoint file.")
parser.add_argument("--data", help = "Input data, a Numpy file.")
parser.add_argument("--class-names", default = "data/modelnet40_ply_hdf5_2048/shape_names.txt", help = "Text file containing a list of class names.")
parser.add_argument("--num-objects", type = int, default = 1000000000, help = "Use the first few objects. Specify a very large number to use all objects.")
args = parser.parse_args()
print(args)

model = importlib.import_module("pointnet_cls")
class_names = [line.rstrip() for line in open(args.class_names)]

with np.load(args.data) as file:
    data_x_original = file["x_original"][:args.num_objects]
    data_x_adv = file["x_adv"][:args.num_objects]

x_pl, _ = model.placeholder_inputs(1, data_x_original.shape[1])

is_training = tf.placeholder(tf.bool, shape = [])

def model_loss_fn(x, t):
    with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
        y, end_points = model.get_model(x, is_training)
    if t is None:
        loss = None
    else:
        loss = model.get_loss(y, t, end_points)
    return y, loss

features_original, features_adv = adversarial_utils.get_feature_vectors(args.checkpoint, x_pl, model_loss_fn, data_x_original, data_x_adv, class_names, extra_feed_dict = {is_training: False})

diff = features_adv - features_original
diff_norm = np.linalg.norm(diff, axis = 1)

print("Average L2 norms of feature vector changes: %s" % np.mean(diff_norm))

percent_pos = np.sum(diff > 0, axis = 1) / float(diff.shape[1])
percent_neg = np.sum(diff < 0, axis = 1) / float(diff.shape[1])

print("Average %% positive changes: %s" % np.mean(percent_pos))
print("Average %% negative changes: %s" % np.mean(percent_neg))