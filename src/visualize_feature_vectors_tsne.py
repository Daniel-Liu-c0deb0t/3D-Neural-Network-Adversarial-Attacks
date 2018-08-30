import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

load_path = ""
load_path_adv = None
out_dir = ""
perplexity = 30

class_names = [line.rstrip() for line in open("shape_names.txt")]

with np.load(load_path) as file:
    feature_vectors = file["feature_vectors"]
    labels = file["labels"]

if load_path_adv is not None:
    with np.load(load_path_adv) as file:
        feature_vectors_adv = file["feature_vectors"]
        labels_adv = file["labels"]

if load_path_adv is None:
    x = feature_vectors
else:
    x = np.concatenate(feature_vectors, feature_vectors_adv)

res = TSNE(n_components = 2, perplexity = perplexity, random_state = 0).fit_transform(x)

if load_path_adv is None:
    embedding = res
else:
    embedding = res[:len(feature_vectors)]
    embedding_adv = res[len(feature_vectors):]

plt.figure(figsize = (15, 15))
plt.subplot(111)

cmap = plt.get_cmap("rainbow")
for i in range(len(class_names)):
    plt.gca().scatter(*embedding[labels == i].T, c = cmap(i / len(class_names)), label = class_names[i])

if load_path_adv is not None:
    plt.gca().scatter(*embedding_adv.T, c = [[0, 0, 0]], label = "adversarial")

plt.gca().legend()
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
plt.savefig(os.path.join(out_dir, "tsne.jpg"))