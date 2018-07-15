import tensorflow as tf

def iter_grad_sign_op(x_pl, model_loss_fn, t_pl = None, one_hot = True, iter = 10, eps = 0.01, clip_min = None, clip_max = None):
    targeted = t_pl is not None
    alpha = eps / float(iter)

    # use the prediction class to prevent label leaking
    if not targeted:
        logits, _ = model_loss_fn(x_pl, None)
        t_pl = tf.argmax(logits, 1)
        if one_hot:
            t_pl = tf.one_hot(t_pl, logits.shape[1])
        t_pl = tf.stop_gradient(t_pl)

    x_adv = x_pl
    for _ in range(iter):
        _, loss = model_loss_fn(x_adv, t_pl)

        if targeted:
            x_adv = x_adv - alpha * tf.sign(tf.gradients(loss, x_adv)[0])
        else:
            x_adv = x_adv + alpha * tf.sign(tf.gradients(loss, x_adv)[0])
    
        if clip_min is not None and clip_max is not None:
            x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)
        
        x_adv = tf.stop_gradient(x_adv)
    
    return x_adv

def momentum_grad_sign_op(x_pl, model_loss_fn, t_pl = None, one_hot = True, iter = 10, eps = 0.01, momentum = 1.0, clip_min = None, clip_max = None):
    targeted = t_pl is not None
    alpha = eps / float(iter)

    # use the prediction class to prevent label leaking
    if not targeted:
        logits, _ = model_loss_fn(x_pl, None)
        t_pl = tf.argmax(logits, 1)
        if one_hot:
            t_pl = tf.one_hot(t_pl, logits.shape[1])
        t_pl = tf.stop_gradient(t_pl)

    x_adv = x_pl
    prev_grad = tf.zeros_like(x_pl)
    for _ in range(iter):
        _, loss = model_loss_fn(x_adv, t_pl)

        grad = tf.gradients(loss, x_adv)[0]
        grad = grad / tf.reduce_mean(tf.abs(grad), axis = list(range(1, x_pl.shape.ndims)), keep_dims = True)
        grad = momentum * prev_grad + grad

        if targeted:
            x_adv = x_adv - alpha * tf.sign(grad)
        else:
            x_adv = x_adv + alpha * tf.sign(grad)
    
        if clip_min is not None and clip_max is not None:
            x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)
        
        x_adv = tf.stop_gradient(x_adv)
    
    return x_adv