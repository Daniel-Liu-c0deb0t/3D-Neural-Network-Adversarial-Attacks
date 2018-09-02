import numpy as np
from collections import defaultdict

file1 = np.load("point_clouds/saliency_original.npz")
file2 = np.load("point_clouds/saliency_adv.npz")

p1 = file1["saliency"][0]
p2 = file2["saliency"][0]

l1 = file1["points"][0]
l2 = file2["points"][0]

print(p1.shape)

s1 = defaultdict(list)
s2 = defaultdict(list)
s3 = []
for i in range(len(l1)):
    s1[l1[i].tobytes()] += [p1[i]]
    s2[l2[i].tobytes()] += [p2[i]]
    if np.any(~np.isclose(l1[i], l2[i])):
        s3.append(p2[i])

print(s3)
print(len(s3))

print(len(s1))
print(len(s2))

print(np.sum(np.all(p1 == 0, axis = 1)))

dup = []
for _, val in s2.items():
    if len(val) > 1:
        dup.append(val)
        print(val)
print(len(dup))

idx1 = np.argsort(np.mean(p1, axis = 1))
print(p1[idx1][:10])

idx2 = np.argsort(np.mean(p2, axis = 1))
print(p2[idx2][:10])