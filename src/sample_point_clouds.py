import point_cloud_utils
import numpy as np

class_names = [line.rstrip() for line in open("shape_names.txt")]
objects, labels = point_cloud_utils.read_off_files("objects/*/test/*.off", class_names)
points, faces = point_cloud_utils.sample_points(objects, 10000)
points, faces = point_cloud_utils.farthest_points(points, faces, 2048)

np.savez_compressed("point_clouds.npz", points = points, faces = faces, labels = labels)