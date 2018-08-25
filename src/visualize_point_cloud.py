import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

load_path = "point_clouds/point_clouds_uneven.npz"
idx = 0
saliency_norm = False
num_points_max = 1024
triangle_mesh = False

saliency = None

class_names = [line.rstrip() for line in open("shape_names.txt")]

if load_path[-3:] == "npy":
    xs, ys, zs = np.load(load_path)[:num_points_max].T
    triangle_mesh = False
elif load_path[-3:] == "npz":
    file = np.load(load_path)

    if "x_adv" in file:
        points, labels = file["x_adv"], file["labels"]
    else:
        points, labels = file["points"], file["labels"]
    
    if "faces" in file:
        faces = file["faces"]
    else:
        print("No triangular faces found in file!")
        triangle_mesh = False
    
    if "saliency" in file:
        saliency = file["saliency"]
        if saliency.ndim == 3:
            saliency = saliency[np.newaxis]
        saliency = saliency[:, idx][:, :num_points_max]
        if "top_k" in file:
            dimension = file["top_k"][:, idx]
        else:
            dimension = None

    print("Label: %s" % class_names[labels[idx]])

    xs, ys, zs = points[idx][:num_points_max].T
    
    if triangle_mesh:
        faces = faces[idx][:num_points_max, :3, [0, 2, 1]]
        unique = np.unique(faces.reshape(-1, faces.shape[-1]), axis = 0)
        triangles = np.empty(shape = (num_points_max, 3))

        for i in range(num_points_max):
            for j in range(3):
                k, = np.where(np.all(unique == faces[i][j], axis = 1))
                triangles[i][j] = k

print("Number of points: %d" % len(xs))

def scale_plot():
    plt.axis("scaled")
    plt.gca().set_xlim(-1, 1)
    plt.gca().set_ylim(-1, 1)
    plt.gca().set_zlim(-1, 1)
    plt.gca().view_init(0, 0)

if saliency is None:
    plt.figure(figsize = (7, 7))
    plt.subplot(111, projection = "3d")
    if triangle_mesh:
        plt.gca().plot_trisurf(*unique.T, triangles = triangles, cmap = "magma")
    plt.gca().scatter(xs, ys, zs, zdir = "y", s = 5)
    scale_plot()
else:
    plt.figure(figsize = (12, 4))
    if saliency_norm:
        saliency = np.linalg.norm(saliency, axis = 2)
        saliency = np.clip(saliency / (np.mean(saliency) * 2.0), 0.0, 1.0)
    else:
        saliency = np.mean(saliency, axis = 2)
        saliency = np.clip(saliency / (np.mean(np.abs(saliency)) * 2.0), -1.0, 1.0) / 2.0 + 0.5
    for i in range(len(saliency)):
        plt.subplot(1, len(saliency), i + 1, projection = "3d")
        if dimension is not None:
            plt.title(dimension[i])
        if triangle_mesh:
            plt.gca().plot_trisurf(*unique.T, triangles = triangles, cmap = "magma")
        plt.gca().scatter(xs, ys, zs, zdir = "y", c = saliency[i], cmap = "viridis_r", s = 5)
        scale_plot()

plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
plt.show()