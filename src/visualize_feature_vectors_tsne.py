import numpy as np
import os
import argparse
import errno
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description = "t-SNE of feature vectors.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("data", help = "Path to the feature vector data file.")
parser.add_argument("--adv", default = None, help = "Path to the adversarial feature vector data file.")
parser.add_argument("--output", default = "feature_vector_tsne", help = "Output directory.")
parser.add_argument("--class-names", default = "data/modelnet40_ply_hdf5_2048/shape_names.txt", help = "Text file containing a list of class names.")
parser.add_argument("--perplexity", default = 30.0, help = "Perplexity for t-SNE.")
args = parser.parse_args()
print(args)

class_names = [line.rstrip() for line in open(args.class_names)]

with np.load(args.data) as file:
    feature_vectors = file["feature_vectors"]
    labels = file["labels"]

if args.adv is not None:
    with np.load(args.adv) as file:
        feature_vectors_adv = file["feature_vectors"]
        labels_adv = file["labels"]

if args.adv is None:
    x = feature_vectors
else:
    x = np.concatenate((feature_vectors, feature_vectors_adv))

res = TSNE(n_components = 2, perplexity = args.perplexity, random_state = 0).fit_transform(x)

if args.adv is None:
    embedding = res
else:
    embedding = res[:len(feature_vectors)]
    embedding_adv = res[len(feature_vectors):]

plt.figure(figsize = (15, 15))
plt.subplot(111)

cmap = plt.get_cmap("rainbow")
for i in range(len(class_names)):
    plt.gca().scatter(*embedding[labels == i].T, c = cmap(float(i) / len(class_names)), label = class_names[i])

if args.adv is not None:
    plt.gca().scatter(*embedding_adv.T, c = [[0.0, 0.0, 0.0]], label = "adversarial")

plt.gca().legend()
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)

try:
    os.makedirs(args.output)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
plt.savefig(os.path.join(args.output, "tsne.jpg"))