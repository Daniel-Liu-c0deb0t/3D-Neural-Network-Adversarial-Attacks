import numpy as np
import tensorflow as tf
import adversarial_utils
import os
import sys
import errno
import argparse
import importlib
working_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(working_dir, "models"))
sys.path.append(os.path.join(working_dir, "utils"))
from collections import defaultdict

parser = argparse.ArgumentParser(description = "Gets the feature vector for PointNet.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--checkpoint", default = "log/model.ckpt", help = "Path to the model's checkpoint file.")
parser.add_argument("--data", help = "Input data, a Numpy file.")
parser.add_argument("--output", default = "feature_vectors", help = "Output path.")
parser.add_argument("--class-names", default = "data/modelnet40_ply_hdf5_2048/shape_names.txt", help = "Text file containing a list of class names.")
parser.add_argument("--num-objects", type = int, default = 1000000000, help = "Use the first few objects. Specify a very large number to use all objects.")
args = parser.parse_args()
print(args)

model = importlib.import_module("pointnet_cls")
class_names = [line.rstrip() for line in open(args.class_names)]

with np.load(args.data) as file:
    data_x_original = file["x_original"][:args.num_objects]
    data_x_adv = file["x_adv"][:args.num_objects]
    labels = file["labels"][:args.num_objects]
    pred_adv = file["pred_adv"][:args.num_objects]

x_pl, _ = model.placeholder_inputs(1, data_x_original.shape[1])

is_training = tf.placeholder(tf.bool, shape = [])

def model_loss_fn(x, t):
    with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
        y, end_points = model.get_model(x, is_training, num_classes = len(class_names))
    if t is None:
        loss = None
    else:
        loss = model.get_loss(y, t, end_points)
    return y, loss

features_original, features_adv, feature_grad_fn, sess_close = adversarial_utils.get_feature_vectors(args.checkpoint, x_pl, model_loss_fn, data_x_original, data_x_adv, class_names, extra_feed_dict = {is_training: False})

print(features_original.shape)
print("Average norm of perturbation: %.3f" % np.mean(np.sqrt(np.sum((data_x_adv - data_x_original) ** 2, axis = (1, 2)))))

avg_ratio = 0.0
ratio_count = 0
for i in range(len(labels)):
    diff_norm = np.linalg.norm(features_adv[i] - features_original[i])
    avg_dist = 0.0
    dist_count = 0
    for j in range(len(labels)):
        if labels[j] == pred_adv[i]:
            dist = np.linalg.norm(features_original[i] - features_original[j])
            avg_dist += dist
            dist_count += 1
    if dist_count == 0:
        continue
    avg_dist /= float(dist_count)
    ratio = diff_norm / avg_dist
    avg_ratio += ratio
    ratio_count += 1

print("Average ratio of difference norms to average class differences: %.3f" % (avg_ratio / float(ratio_count)))

def print_per_class(per_class):
    for i, val in enumerate(per_class):
        print("%d, %s: %.3f" % (i, class_names[i], val))

counts = np.zeros(len(class_names))
np.add.at(counts, labels, 1)
zero = counts == 0.0
counts[zero] = 1.0

diff = features_adv - features_original
diff_norm = np.linalg.norm(diff, axis = 1)

print("Average L2 norms of feature vector changes: %.3f" % np.mean(diff_norm))

per_class = np.zeros(len(class_names))
np.add.at(per_class, labels, diff_norm)
per_class[zero] = 0.0
per_class = per_class / counts

print("Average L2 norms of feature vector changes per class:")
print_per_class(per_class)

norm_diff = np.linalg.norm(features_adv, axis = 1) - np.linalg.norm(features_original, axis = 1)

print("Average difference between L2 norms: %.3f" % np.mean(norm_diff))
print("Min difference between L2 norms: %.3f" % np.min(norm_diff))
print("Max difference between L2 norms: %.3f" % np.max(norm_diff))

per_class = np.zeros(len(class_names))
np.add.at(per_class, labels, norm_diff)
per_class[zero] = 0.0
per_class = per_class / counts

print("Average difference between L2 norms per class:")
print_per_class(per_class)

percent_pos = np.sum(diff > 0.0, axis = 1) / float(diff.shape[1])
percent_neg = np.sum(diff < 0.0, axis = 1) / float(diff.shape[1])

print("Average %% positive change: %.3f" % np.mean(percent_pos))
print("Average %% negative change: %.3f" % np.mean(percent_neg))

per_class_pos = np.zeros(len(class_names))
np.add.at(per_class_pos, labels, percent_pos)
per_class_pos[zero] = 0.0
per_class_pos = per_class_pos / counts

per_class_neg = np.zeros(len(class_names))
np.add.at(per_class_neg, labels, percent_neg)
per_class_neg[zero] = 0.0
per_class_neg = per_class_neg / counts

print("Average % positive change per class:")
print_per_class(per_class_pos)

print("Average % negative change per class:")
print_per_class(per_class_neg)

print("Average dimension change %.3f" % np.mean(diff))

per_class = np.zeros(len(class_names))
np.add.at(per_class, labels, np.mean(diff, axis = 1))
per_class[zero] = 0.0
per_class = per_class / counts

print("Average dimension change per class:")
print_per_class(per_class)

print("Average absolute dimension change %.3f" % np.mean(np.abs(diff)))
print("Avg min absolute dimension change %.3f" % np.min(np.mean(np.abs(diff), axis = 1)))
print("Avg max absolute dimension change %.3f" % np.max(np.mean(np.abs(diff), axis = 1)))

per_class = np.zeros(len(class_names))
np.add.at(per_class, labels, np.mean(np.abs(diff), axis = 1))
per_class[zero] = 0.0
per_class = per_class / counts

print("Average absolute dimension change per class:")
print_per_class(per_class)

avg_diff = np.mean(diff, axis = 0)
print("Average change per dimension, min %.3f, max %.3f" % (np.min(avg_diff), np.max(avg_diff)))

pair_dist_original = np.zeros((len(class_names), len(class_names)))
pair_dist_adv = np.zeros((len(class_names), len(class_names)))
pair_dist_pred_adv_original = np.zeros((len(class_names), len(class_names)))
pair_dist_pred_adv = np.zeros((len(class_names), len(class_names)))
pair_counts = np.zeros((len(class_names), len(class_names)))
pair_counts_pred_adv = np.zeros((len(class_names), len(class_names)))
mask = np.zeros((len(class_names), len(class_names)))
for i in range(len(labels)):
    for j in range(len(labels)):
        pair_dist_original[labels[i]][labels[j]] += np.linalg.norm(features_original[i] - features_original[j])
        pair_dist_adv[labels[i]][labels[j]] += np.linalg.norm(features_adv[i] - features_original[j])
        pair_dist_pred_adv_original[pred_adv[i]][labels[j]] += np.linalg.norm(features_original[i] - features_original[j])
        pair_dist_pred_adv[pred_adv[i]][labels[j]] += np.linalg.norm(features_adv[i] - features_original[j])
        pair_counts[labels[i]][labels[j]] += 1
        pair_counts_pred_adv[pred_adv[i]][labels[j]] += 1

pair_dist_original = np.array(pair_dist_original)
pair_dist_adv = np.array(pair_dist_adv)
pair_counts = np.array(pair_counts)
pair_dist_pred_adv_original = np.array(pair_dist_pred_adv_original)
pair_dist_pred_adv = np.array(pair_dist_pred_adv)
pair_counts_pred_adv = np.array(pair_counts_pred_adv)
mask[labels, pred_adv] = 1.0

pair_dist_original[pair_counts == 0.0] = 0.0
pair_dist_adv[pair_counts == 0.0] = 0.0
pair_dist_pred_adv_original[pair_counts_pred_adv == 0.0] = 0.0
pair_dist_pred_adv[pair_counts_pred_adv == 0.0] = 0.0
pair_counts[pair_counts == 0.0] = 1.0
pair_counts_pred_adv[pair_counts_pred_adv == 0.0] = 1.0

pair_dist_original = pair_dist_original / pair_counts
pair_dist_adv = pair_dist_adv / pair_counts
pair_dist_pred_adv_original = pair_dist_pred_adv_original / pair_counts_pred_adv
pair_dist_pred_adv = pair_dist_pred_adv / pair_counts_pred_adv

try:
    os.makedirs(args.output)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

adversarial_utils.heatmap(pair_dist_original, os.path.join(args.output, "original_dist.png"), "Original", "Original", class_names = class_names, percentages = False, annotate = False)
adversarial_utils.heatmap(pair_dist_adv, os.path.join(args.output, "adversarial_dist.png"), "Adversarial", "Original", class_names = class_names, percentages = False, annotate = False)
adversarial_utils.heatmap(pair_dist_adv - pair_dist_original, os.path.join(args.output, "adversarial_dist_change.png"), "Adversarial", "Original", class_names = class_names, percentages = False, annotate = False)
adversarial_utils.heatmap((pair_dist_adv - pair_dist_original) * mask, os.path.join(args.output, "adversarial_dist_relevant_change.png"), "Adversarial", "Original", class_names = class_names, percentages = False, annotate = False)

adversarial_utils.heatmap(pair_dist_pred_adv_original, os.path.join(args.output, "original_pred_adv_dist.png"), "Original", "Original", class_names = class_names, percentages = False, annotate = False)
adversarial_utils.heatmap(pair_dist_pred_adv, os.path.join(args.output, "adversarial_pred_adv_dist.png"), "Adversarial", "Original", class_names = class_names, percentages = False, annotate = False)
adversarial_utils.heatmap(pair_dist_pred_adv - pair_dist_pred_adv_original, os.path.join(args.output, "adversarial_pred_adv_dist_change.png"), "Adversarial", "Original", class_names = class_names, percentages = False, annotate = False)

def top_k_freq(a, k):
    res = defaultdict(int)
    max_idx = np.argsort(np.abs(a), axis = 1)[:, -k:]
    for idx in max_idx:
        for i in idx:
            res[i] += 1
    return sorted(res.items(), key = lambda x: x[1], reverse = True)

def top_k_freq_per_class(a, k):
    res = [defaultdict(int) for _ in class_names]
    max_idx = np.argsort(np.abs(a), axis = 1)[:, -k:]
    for i in range(len(max_idx)):
        for j in max_idx[i]:
            res[labels[i]][j] += 1
    return [sorted(dict.items(), key = lambda x: x[1], reverse = True) for dict in res]

print("Dimensions that changed the most (dimension, count):")
freq = top_k_freq(diff, 5)[:5]
print(freq)

freq_per_class = top_k_freq_per_class(diff, 5)

print("Number of different dimensions that changed the most per class:")
print_per_class([len(x) for x in freq_per_class])

print("Dimensions that changed the most per class (dimension, count):")
for i in range(len(class_names)):
    curr_idx = freq_per_class[i][:5]
    print("%d, %s: %s" % (i, class_names[i], curr_idx))

grads_adv, top_adv = feature_grad_fn(diff, k = 5, adv = True)
grads_original, top_original = feature_grad_fn(diff, k = 5, adv = False)
sess_close()

np.savez_compressed(os.path.join(args.output, "feature_vector_saliency_adv.npz"), feature_vectors = features_adv, points = data_x_adv, labels = labels, pred_adv = pred_adv, saliency = grads_adv, top_k = top_adv)
np.savez_compressed(os.path.join(args.output, "feature_vector_saliency_original.npz"), feature_vectors = features_original, points = data_x_original, labels = labels, saliency = grads_original, top_k = top_original)