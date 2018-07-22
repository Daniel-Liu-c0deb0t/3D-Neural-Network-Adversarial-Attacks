import glob
import os
import numpy as np
import bisect

def read_off_files(globPath, label_names = None):
    if label_names is not None:
        label_names = {label_names[i]: i for i in range(len(label_names))}

    objects = []
    labels = []
    for path in glob.glob(globPath):
        with open(path) as file:
            line = file.readline()
            if len(line) > 4:
                line = line[3:] # exclude 'OFF'
            else:
                line = file.readline()
            num_vertices, num_faces, _ = [int(x) for x in line.split()]

            vertices = []
            for _ in range(num_vertices):
                vertices.append([float(x) for x in file.readline().split()])
            
            faces = []
            for _ in range(num_faces):
                curr_face = []
                idx = [int(x) for x in file.readline().split()]
                idx = idx[1:]
                for i in idx:
                    curr_face.append(vertices[i])
                faces.append(curr_face)
            
            faces = np.array(faces) # the shape should be (num_faces, 3, 3)
            if tuple(faces.shape) == (num_faces, 3, 3):
                name = os.path.basename(path)
                name = name[:name.rindex("_")]
                if label_names is None or name in label_names:
                    if label_names is not None:
                        name = label_names[name]
                    objects.append(faces[:, :, [0, 2, 1]])
                    labels.append(name)
                else:
                    raise ValueError("A label does not exist in label names!")
            else:
                raise ValueError("A 3D object's array has incorrect shape!")
    
    return objects, np.array(labels)

def sample_points(objects, num_points):
    points = []
    triangles = []

    for obj in objects:
        curr_points = []
        curr_triangles = []

        areas = np.cross(obj[:, 1] - obj[:, 0], obj[:, 2] - obj[:, 0])
        areas = np.linalg.norm(areas, axis = 1) / 2.0
        prefix_sum = np.cumsum(areas)
        total_area = prefix_sum[-1]
        
        for _ in range(num_points):
            # pick random triangle based on area
            rand = np.random.uniform(high = total_area)
            if rand >= total_area:
                idx = len(obj) - 1 # can happen due to floating point rounding
            else:
                idx = bisect.bisect_right(prefix_sum, rand)
            
            # pick random point in triangle
            a, b, c = obj[idx]
            r1 = np.random.random()
            r2 = np.random.random()
            if r1 + r2 >= 1.0:
                r1 = 1 - r1
                r2 = 1 - r2
            p = a + r1 * (b - a) + r2 * (c - a)

            curr_points.append(p)
            curr_triangles.append(obj[idx])

        points.append(curr_points)
        triangles.append(curr_triangles)
    
    points = np.array(points)
    triangles = np.array(triangles)

    return points, triangles

def farthest_points_normalized(points, faces, num_points):
    res_points = []
    res_faces = []

    for obj_points, obj_faces in zip(points, faces):
        first = np.random.randint(len(obj_points))
        selected = [first]
        dists = np.full(shape = len(obj_points), fill_value = np.inf)

        for _ in range(num_points - 1):
            dists = np.minimum(dists, np.linalg.norm(obj_points - obj_points[selected[-1]][np.newaxis, :], axis = 1))
            selected.append(np.argmax(dists))
        
        res_points.append(obj_points[selected])
        res_faces.append(obj_faces[selected])
    
    res_points = np.array(res_points)
    res_faces = np.array(res_faces)

    # normalize the points and faces
    avg = np.average(np.transpose(res_points, axes = (0, 2, 1)), axis = 2)
    res_points = res_points - avg[:, np.newaxis, :]
    res_faces = res_faces - avg[:, np.newaxis, np.newaxis, :]
    dists = np.max(np.linalg.norm(res_points, axis = 2), axis = 1)
    res_points = res_points / dists[:, np.newaxis, np.newaxis]
    res_faces = res_faces / dists[:, np.newaxis, np.newaxis, np.newaxis]

    return res_points, res_faces