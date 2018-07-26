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

parser = argparse.ArgumentParser(description = "Evaluates Pointnet++ on classification.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--checkpoint", default = "log/model.ckpt", help = "Path to the model's checkpoint file.")
parser.add_argument("--output", default = "evaluate", help = "Output directory.")
parser.add_argument("--data", default = "data/modelnet40_ply_hdf5_2048/test_files.txt", help = "Input data. Either a Numpy file or a text file containing a list of HDF5 files.")
parser.add_argument("--class-names", default = "data/modelnet40_ply_hdf5_2048/shape_names.txt", help = "Text file containing a list of class names.")
parser.add_argument("--numpy", action = "store_true", help = "Use the data file as a numpy file.")
parser.add_argument("--num-points", type = int, default = 1024, help = "Number of points to use.")
parser.add_argument("--num-objects", type = int, default = 1000000000, help = "Use the first few objects. Specify a very large number to use all objects.")
args = parser.parse_args()
print(args)

model = importlib.import_module("pointnet2_cls_ssg")
class_names = [line.rstrip() for line in open(args.class_names)]

if args.numpy:
    file = np.load(args.data)
    data_x = file["points"][:, :args.num_points, :]
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
    data_t = np.concatenate(data_t)

data_x = data_x[:args.num_objects]
data_t = data_t[:args.num_objects]

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

adversarial_utils.evaluate(args.checkpoint, args.output, x_pl, t_pl, model_loss_fn, data_x, data_t, class_names, one_hot = False, extra_feed_dict = {is_training: False})