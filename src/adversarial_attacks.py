import tensorflow as tf

def iter_grad_sign_op(x_pl, model_loss_fn, t_pl = None, faces = None, one_hot = True, iter = 10, eps = 0.01, restrict = False, clip_min = None, clip_max = None):
    targeted = t_pl is not None
    alpha = eps / float(iter)

    # use the prediction class to prevent label leaking
    if not targeted:
        logits, _ = model_loss_fn(x_pl, None)
        t_pl = tf.argmax(logits, 1)
        if one_hot:
            t_pl = tf.one_hot(t_pl, tf.shape(logits)[1])
        t_pl = tf.stop_gradient(t_pl)

    if faces is not None:
        normal = tf.cross(faces[:, :, 1] - faces[:, :, 0], faces[:, :, 2] - faces[:, :, 1])
        normal = normal / tf.norm(normal, axis = 2, keep_dims = True)

    x_adv = x_pl
    for _ in range(iter):
        _, loss = model_loss_fn(x_adv, t_pl)

        x_original = x_adv

        if targeted:
            x_adv = x_adv - alpha * tf.sign(tf.gradients(loss, x_adv)[0])
        else:
            x_adv = x_adv + alpha * tf.sign(tf.gradients(loss, x_adv)[0])

        if faces is not None:
            # constrain perturbations for each point to its corresponding plane
            x_adv = x_adv - normal * tf.reduce_sum(normal * (x_adv - faces[:, :, 0]), axis = 2, keep_dims = True)
            # clip perturbations that goes outside each triangle
            if restrict:
                x_adv = triangle_border_intersections_op(x_original, x_adv, faces)

        if clip_min is not None and clip_max is not None:
            x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)
        
        x_adv = tf.stop_gradient(x_adv)
    
    return x_adv

def momentum_grad_sign_op(x_pl, model_loss_fn, t_pl = None, faces = None, one_hot = True, iter = 10, eps = 0.01, momentum = 1.0, restrict = False, clip_min = None, clip_max = None):
    targeted = t_pl is not None
    alpha = eps / float(iter)

    # use the prediction class to prevent label leaking
    if not targeted:
        logits, _ = model_loss_fn(x_pl, None)
        t_pl = tf.argmax(logits, 1)
        if one_hot:
            t_pl = tf.one_hot(t_pl, tf.shape(logits)[1])
        t_pl = tf.stop_gradient(t_pl)

    if faces is not None:
        normal = tf.cross(faces[:, :, 1] - faces[:, :, 0], faces[:, :, 2] - faces[:, :, 1])
        normal = normal / tf.norm(normal, axis = 2, keep_dims = True)

    x_adv = x_pl
    prev_grad = tf.zeros_like(x_pl)
    for _ in range(iter):
        _, loss = model_loss_fn(x_adv, t_pl)

        grad = tf.gradients(loss, x_adv)[0]
        grad = grad / tf.reduce_mean(tf.abs(grad), axis = list(range(1, x_pl.shape.ndims)), keep_dims = True)
        grad = momentum * prev_grad + grad

        x_original = x_adv

        if targeted:
            x_adv = x_adv - alpha * tf.sign(grad)
        else:
            x_adv = x_adv + alpha * tf.sign(grad)

        if faces is not None:
            # constrain perturbations for each point to its corresponding plane
            x_adv = x_adv - normal * tf.reduce_sum(normal * (x_adv - faces[:, :, 0]), axis = 2, keep_dims = True)
            # clip perturbations that goes outside each triangle
            if restrict:
                x_adv = triangle_border_intersections_op(x_original, x_adv, faces)

        if clip_min is not None and clip_max is not None:
            x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)
        
        x_adv = tf.stop_gradient(x_adv)
    
    return x_adv

inf = float("inf")
float_epsilon = 1e-4 # to handle floating point inaccuracies

def triangle_border_intersections_op(p1, p2, triangles):
    p = p1
    d = p2 - p1

    triangle_normals = tf.cross(triangles[:, :, 1] - triangles[:, :, 0], triangles[:, :, 2] - triangles[:, :, 1])

    side_normals = tf.stack([
        tf.cross(triangles[:, :, 1] - triangles[:, :, 0], triangles[:, :, 1] - triangles[:, :, 0] + triangle_normals),
        tf.cross(triangles[:, :, 2] - triangles[:, :, 1], triangles[:, :, 2] - triangles[:, :, 1] + triangle_normals),
        tf.cross(triangles[:, :, 0] - triangles[:, :, 2], triangles[:, :, 0] - triangles[:, :, 2] + triangle_normals)
    ], axis = 2)

    # intersection between line and triangle sides as planes
    dot = tf.reduce_sum(side_normals * d[:, :, tf.newaxis, :], axis = 3)
    zero_mask = tf.equal(dot, 0)
    dot = tf.where(zero_mask, tf.ones_like(dot), dot) # prevent division by zero
    a = p[:, :, tf.newaxis, :] - triangles
    b = -tf.reduce_sum(side_normals * a, axis = 3) / dot
    dir = d[:, :, tf.newaxis, :] * b[:, :, :, tf.newaxis]
    # intersections not in the same direction as d have b < 0
    mask = tf.logical_or(zero_mask, b < -float_epsilon)[:, :, :, tf.newaxis] & tf.fill(tf.shape(dir), True)
    dir = tf.where(mask, tf.fill(tf.shape(dir), inf), dir)

    # only use closest intersection
    dists = tf.linalg.norm(dir, axis = 3)
    min_idx = tf.argmin(dists, axis = 2)
    closest_mask = tf.one_hot(min_idx, 3, on_value = True, off_value = False)[:, :, :, tf.newaxis] & tf.fill(tf.shape(dir), True)
    dir = tf.where(closest_mask, dir, tf.zeros_like(dir))
    dir = tf.reduce_sum(dir, axis = 2)
    # either use the intersection point or d
    dists = tf.linalg.norm(dir, axis = 2)
    norm_d = tf.linalg.norm(d, axis = 2)
    closest_mask = (norm_d < dists)[:, :, tf.newaxis] & tf.fill(tf.shape(dir), True)
    dir = tf.where(closest_mask, d, dir)

    return p + dir