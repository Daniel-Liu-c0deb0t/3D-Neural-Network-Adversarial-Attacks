import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import adversarial_attacks
import os
import errno

def confusion_heatmap(data, path, class_names = None, percentages = True, annotate = True):
    data = np.array(data)
    
    if percentages:
        total = np.sum(data, axis = 1, keepdims = True)
        total[total == 0] = 1 # handle division by zero
        data = data.astype(float) / total

    heatmap(data, path, "Target Classes", "Predicted Classes", class_names = class_names, percentages = percentages, annotate = annotate)

def class_change_heatmap(data, path, class_names = None, percentages = True, annotate = True):
    data = np.array(data)
    
    if percentages:
        total = np.sum(data, axis = 1, keepdims = True)
        total[total == 0] = 1 # handle division by zero
        data = data.astype(float) / total

    heatmap(data, path, "Original Classes", "Classes After Adversarial Attack", class_names = class_names, percentages = percentages, annotate = annotate)

def targeted_success_rate_heatmap(data, path, total = None, class_names = None, annotate = True):
    data = np.array(data)
    percentages = total is not None

    if percentages:
        total = np.array(total)
        total[total == 0] = 1 # handle division by zero
        data = data.astype(float) / total
    
    heatmap(data, path, "Original Classes", "Adversarial Attack Target Classes", class_names = class_names, percentages = percentages, annotate = annotate)

def heatmap(data, path, x_label, y_label, class_names = None, percentages = True, annotate = True):
    data = np.array(data)

    if class_names is None:
        class_names = list(range(len(data)))
    
    if percentages:
        vmin = 0
        vmax = 1
        fmt = ".2g"
    else:
        vmin = None
        vmax = None
        fmt = "d"
    
    fig_size = len(class_names) // 10 * 7
    plt.figure(figsize = (fig_size, fig_size))
    ax = sns.heatmap(data.T, annot = annotate, xticklabels = class_names, yticklabels = class_names, vmin = vmin, vmax = vmax, fmt = fmt, linewidths = 2)
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.savefig(path)
    plt.close()

def untargeted_attack(model_path, out_dir, x_pl, t_pl, model_loss_fn, data_x, data_t, class_names, iter, eps_list, data_f = None, one_hot = True, use_momentum = False, momentum = 1.0, clip_min = None, clip_max = None, extra_feed_dict = None):
    if extra_feed_dict is None:
        extra_feed_dict = {}
    try:
        os.makedirs(out_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    logits_op, loss_op = model_loss_fn(x_pl, t_pl)

    data_x = np.array(data_x)
    data_t = np.array(data_t)
    if data_f is not None:
        data_f = np.array(data_f)
    eps_list = np.array(eps_list)

    eps = tf.placeholder(tf.float32, shape = [])
    if data_f is None:
        faces = None
    else:
        faces = tf.placeholder(tf.float32, shape = [1, None, 3, 3])

    if use_momentum:
        x_adv_op = adversarial_attacks.momentum_grad_sign_op(x_pl, model_loss_fn, faces = faces, one_hot = one_hot, iter = iter, eps = eps, momentum = momentum, clip_min = clip_min, clip_max = clip_max)
    else:
        x_adv_op = adversarial_attacks.iter_grad_sign_op(x_pl, model_loss_fn, faces = faces, one_hot = one_hot, iter = iter, eps = eps, clip_min = clip_min, clip_max = clip_max)
    
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        saver.restore(sess, model_path)
        print("Restored model!")

        succeeded_x_original = []
        succeeded_target = []
        succeeded_x_adv = []
        succeeded_pred_adv = []
        if data_f is not None:
            succeeded_faces = []

        total = len(data_x)

        logits = []
        losses = []
        preds = []
        for i in range(total):
            feed_dict = {
                x_pl: [data_x[i]],
                t_pl: [data_t[i]]
            }
            feed_dict.update(extra_feed_dict)
            curr_logit, curr_loss = sess.run([logits_op, loss_op], feed_dict = feed_dict)
            curr_pred = np.argmax(curr_logit, axis = 1)
            logits.append(curr_logit)
            losses.append(curr_loss)
            preds.append(curr_pred)
        
        logits = np.concatenate(logits)
        losses = np.array(losses)
        preds = np.concatenate(preds)

        if one_hot:
            sparse_t = np.argmax(data_t, axis = 1)
        else:
            sparse_t = data_t
        
        correct_idx = preds == sparse_t
        logits = logits[correct_idx]
        preds = preds[correct_idx]
        losses = losses[correct_idx]
        data_x = data_x[correct_idx]
        data_t = data_t[correct_idx]
        if data_f is not None:
            data_f = data_f[correct_idx]

        correct = len(data_x)

        class_totals = np.zeros(shape = len(class_names), dtype = int)
        np.add.at(class_totals, sparse_t, 1)
        class_correct = np.zeros(shape = len(class_names), dtype = int)
        np.add.at(class_correct, preds, 1)

        print("Evaluated model!")
        print("Generating adversarial inputs...")

        for curr_eps in eps_list:
            print("Current eps: %s" % curr_eps)

            x_adv = []
            for i in range(correct):
                feed_dict = {
                    x_pl: [data_x[i]],
                    eps: curr_eps
                }
                if data_f is not None:
                    feed_dict[faces] = [data_f[i]]
                feed_dict.update(extra_feed_dict)
                curr_x_adv = sess.run(x_adv_op, feed_dict = feed_dict)
                x_adv.append(curr_x_adv)
            
            x_adv = np.concatenate(x_adv)

            logits_adv = []
            losses_adv = []
            preds_adv = []
            for i in range(correct):
                feed_dict = {
                    x_pl: [x_adv[i]],
                    t_pl: [data_t[i]]
                }
                feed_dict.update(extra_feed_dict)
                curr_logit_adv, curr_loss_adv = sess.run([logits_op, loss_op], feed_dict = feed_dict)
                curr_pred_adv = np.argmax(curr_logit_adv, axis = 1)
                logits_adv.append(curr_logit_adv)
                losses_adv.append(curr_loss_adv)
                preds_adv.append(curr_pred_adv)
            
            logits_adv = np.concatenate(logits_adv)
            losses_adv = np.array(losses_adv)
            preds_adv = np.concatenate(preds_adv)

            succeeded_idx = preds_adv != preds
            succeeded = np.sum(succeeded_idx)
            succeeded_x_original.append(data_x[succeeded_idx])
            succeeded_target.append(data_t[succeeded_idx])
            succeeded_x_adv.append(x_adv[succeeded_idx])
            succeeded_pred_adv.append(preds_adv[succeeded_idx])
            if data_f is not None:
                succeeded_faces.append(data_f[succeeded_idx])
            
            class_changes = np.zeros(shape = (len(class_names), len(class_names)), dtype = int)
            np.add.at(class_changes, [preds, preds_adv], 1)

            eps_str = str(curr_eps).replace(".", "_")
            class_change_heatmap(class_changes, os.path.join(out_dir, "class_changes_eps_%s.png" % eps_str), class_names = class_names, percentages = False)
            class_change_heatmap(class_changes, os.path.join(out_dir, "percent_class_changes_eps_%s.png" % eps_str), class_names = class_names, annotate = False)

            class_succeeded = np.zeros(shape = len(class_names), dtype = int)
            np.add.at(class_succeeded, preds[succeeded_idx], 1)

            with open(os.path.join(out_dir, "class_stats_eps_%s.csv" % eps_str), "w") as f:
                f.write("Index, Original Class, Total, Correct, Attack Succeeded, Succeeded / Correct\n")

                for i in range(len(class_names)):
                    percent = 0 if class_correct[i] == 0 else float(class_succeeded[i]) / class_correct[i]
                    f.write("%d, %s, %d, %d, %d, %.3f\n" % (i, class_names[i], class_totals[i], class_correct[i], class_succeeded[i], percent))
                
                percent = 0 if correct == 0 else float(succeeded) / correct
                f.write("Total, Total, %d, %d, %d, %.3f\n" % (total, correct, succeeded, percent))

            if data_f is None:
                np.savez_compressed(os.path.join(out_dir, "succeeded_point_clouds_eps_%s.npz" % eps_str), x_original = succeeded_x_original[-1], labels = succeeded_target[-1], x_adv = succeeded_x_adv[-1], pred_adv = succeeded_pred_adv[-1])
            else:
                np.savez_compressed(os.path.join(out_dir, "succeeded_point_clouds_eps_%s.npz" % eps_str), x_original = succeeded_x_original[-1], labels = succeeded_target[-1], x_adv = succeeded_x_adv[-1], pred_adv = succeeded_pred_adv[-1], faces = succeeded_faces[-1])

    print("Done!")

    if data_f is None:
        return succeeded_x_original, succeeded_target, succeeded_x_adv, succeeded_pred_adv
    else:
        return succeeded_x_original, succeeded_target, succeeded_x_adv, succeeded_pred_adv, succeeded_faces

def targeted_attack(model_path, out_dir, x_pl, t_pl, model_loss_fn, data_x, data_t, class_names, iter, eps_list, data_f = None, one_hot = True, use_momentum = False, momentum = 1.0, clip_min = None, clip_max = None, extra_feed_dict = None):
    if extra_feed_dict is None:
        extra_feed_dict = {}
    try:
        os.makedirs(out_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    logits_op, loss_op = model_loss_fn(x_pl, t_pl)

    data_x = np.array(data_x)
    data_t = np.array(data_t)
    if data_f is not None:
        data_f = np.array(data_f)
    eps_list = np.array(eps_list)

    eps = tf.placeholder(tf.float32, shape = [])
    if one_hot:
        target = tf.placeholder(tf.float32, shape = [1, len(class_names)])
    else:
        target = tf.placeholder(tf.int32, [1])
    if data_f is None:
        faces = None
    else:
        faces = tf.placeholder(tf.float32, shape = [1, None, 3, 3])
    
    if use_momentum:
        x_adv_op = adversarial_attacks.momentum_grad_sign_op(x_pl, model_loss_fn, t_pl = target, faces = faces, one_hot = one_hot, iter = iter, eps = eps, momentum = momentum, clip_min = clip_min, clip_max = clip_max)
    else:
        x_adv_op = adversarial_attacks.iter_grad_sign_op(x_pl, model_loss_fn, t_pl = target, faces = faces, one_hot = one_hot, iter = iter, eps = eps, clip_min = clip_min, clip_max = clip_max)
    
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        saver.restore(sess, model_path)
        print("Model restored!")
        
        succeeded_x_original = []
        succeeded_target = []
        succeeded_x_adv = []
        if data_f is not None:
            succeeded_faces = []

        total = len(data_x)

        logits = []
        losses = []
        preds = []
        for i in range(total):
            feed_dict = {
                x_pl: [data_x[i]],
                t_pl: [data_t[i]]
            }
            feed_dict.update(extra_feed_dict)
            curr_logit, curr_loss = sess.run([logits_op, loss_op], feed_dict = feed_dict)
            curr_pred = np.argmax(curr_logit, axis = 1)
            logits.append(curr_logit)
            losses.append(curr_loss)
            preds.append(curr_pred)
        
        logits = np.concatenate(logits)
        losses = np.array(losses)
        preds = np.concatenate(preds)

        if one_hot:
            sparse_t = np.argmax(data_t, axis = 1)
        else:
            sparse_t = data_t
        
        correct_idx = preds == sparse_t
        logits = logits[correct_idx]
        preds = preds[correct_idx]
        losses = losses[correct_idx]
        data_x = data_x[correct_idx]
        data_t = data_t[correct_idx]
        if data_f is not None:
            data_f = data_f[correct_idx]

        correct = len(data_x)

        class_totals = np.zeros(shape = len(class_names), dtype = int)
        np.add.at(class_totals, sparse_t, 1)
        class_correct = np.zeros(shape = len(class_names), dtype = int)
        np.add.at(class_correct, preds, 1)
        heatmap_totals = np.tile(class_correct, (len(class_names), 1)).T

        print("Model evaluated!")
        print("Generating adversarial inputs...")

        for curr_eps in eps_list:
            print("Current eps: %s" % curr_eps)

            curr_succeeded_x_original = []
            curr_succeeded_target = []
            curr_succeeded_x_adv = []
            if data_f is not None:
                curr_succeeded_faces = []
            success_counts = np.zeros(shape = (len(class_names), len(class_names)), dtype = int)
            total_succeeded = 0
            eps_str = str(curr_eps).replace(".", "_")

            for curr_target in range(len(class_names)):
                print("Current target: %s" % class_names[curr_target])

                if one_hot:
                    adv_target = np.zeros(shape = len(class_names))
                    adv_target[curr_target] = 1
                else:
                    adv_target = curr_target
                
                x_adv = []
                for i in range(correct):
                    feed_dict = {
                        x_pl: [data_x[i]],
                        eps: curr_eps,
                        target: [adv_target]
                    }
                    if data_f is not None:
                        feed_dict[faces] = [data_f[i]]
                    feed_dict.update(extra_feed_dict)
                    curr_x_adv = sess.run(x_adv_op, feed_dict = feed_dict)
                    x_adv.append(curr_x_adv)
                
                x_adv = np.concatenate(x_adv)

                logits_adv = []
                losses_adv = []
                preds_adv = []
                for i in range(correct):
                    feed_dict = {
                        x_pl: [x_adv[i]],
                        t_pl: [data_t[i]]
                    }
                    feed_dict.update(extra_feed_dict)
                    curr_logit_adv, curr_loss_adv = sess.run([logits_op, loss_op], feed_dict = feed_dict)
                    curr_pred_adv = np.argmax(curr_logit_adv, axis = 1)
                    logits_adv.append(curr_logit_adv)
                    losses_adv.append(curr_loss_adv)
                    preds_adv.append(curr_pred_adv)
                
                logits_adv = np.concatenate(logits_adv)
                losses_adv = np.array(losses_adv)
                preds_adv = np.concatenate(preds_adv)

                succeeded_idx = preds_adv == np.repeat(curr_target, len(data_x))
                succeeded = np.sum(succeeded_idx)
                total_succeeded += succeeded
                curr_succeeded_x_original.append(data_x[succeeded_idx])
                curr_succeeded_target.append(data_t[succeeded_idx])
                curr_succeeded_x_adv.append(x_adv[succeeded_idx])
                if data_f is not None:
                    curr_succeeded_faces.append(data_f[succeeded_idx])
                
                np.add.at(success_counts, [preds[succeeded_idx], preds_adv[succeeded_idx]], 1)

                if data_f is None:
                    np.savez_compressed(os.path.join(out_dir, "succeeded_point_clouds_target_%s_eps_%s.npz" % (class_names[curr_target], eps_str)), x_original = curr_succeeded_x_original[-1], labels = curr_succeeded_target[-1], x_adv = curr_succeeded_x_adv[-1])
                else:
                    np.savez_compressed(os.path.join(out_dir, "succeeded_point_clouds_target_%s_eps_%s.npz" % (class_names[curr_target], eps_str)), x_original = curr_succeeded_x_original[-1], labels = curr_succeeded_target[-1], x_adv = curr_succeeded_x_adv[-1], faces = curr_succeeded_faces[-1])

            targeted_success_rate_heatmap(success_counts, os.path.join(out_dir, "success_count_eps_%s.png" % eps_str), class_names = class_names)
            targeted_success_rate_heatmap(success_counts, os.path.join(out_dir, "success_rate_eps_%s.png" % eps_str), total = heatmap_totals, class_names = class_names, annotate = False)

            total_succeeded /= len(class_names)

            with open(os.path.join(out_dir, "targeted_stats_eps_%s.csv" % eps_str), "w") as f:
                percent = 0 if correct == 0 else float(total_succeeded) / correct
                f.write("Total %d, Correct %d, Average Attack Succeeded Per Target Class %d, Average Succeeded / Correct %.3f\n" % (total, correct, total_succeeded, percent))
            
            succeeded_x_original.append(curr_succeeded_x_original)
            succeeded_target.append(curr_succeeded_target)
            succeeded_x_adv.append(curr_succeeded_x_adv)
            if data_f is not None:
                succeeded_faces.append(curr_succeeded_faces)

    print("Done!")

    if data_f is None:
        return succeeded_x_original, succeeded_target, succeeded_x_adv
    else:
        return succeeded_x_original, succeeded_target, succeeded_x_adv, succeeded_faces

def evaluate(model_path, out_dir, x_pl, t_pl, model_loss_fn, data_x, data_t, class_names, one_hot = True, extra_feed_dict = None):
    if extra_feed_dict is None:
        extra_feed_dict = {}
    try:
        os.makedirs(out_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    logits_op, loss_op = model_loss_fn(x_pl, t_pl)

    data_x = np.array(data_x)
    data_t = np.array(data_t)

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        saver.restore(sess, model_path)
        print("Model restored!")

        logits = []
        losses = []
        preds = []
        for i in range(len(data_x)):
            feed_dict = {
                x_pl: [data_x[i]],
                t_pl: [data_t[i]]
            }
            feed_dict.update(extra_feed_dict)
            curr_logit, curr_loss = sess.run([logits_op, loss_op], feed_dict = feed_dict)
            curr_pred = np.argmax(curr_logit, axis = 1)
            logits.append(curr_logit)
            losses.append(curr_loss)
            preds.append(curr_pred)
        
        logits = np.concatenate(logits)
        losses = np.array(losses)
        preds = np.concatenate(preds)

        if one_hot:
            sparse_t = np.argmax(data_t, axis = 1)
        else:
            sparse_t = data_t
        
        correct = np.sum(preds == sparse_t)

        target_vs_preds = np.zeros(shape = (len(class_names), len(class_names)), dtype = int)
        np.add.at(target_vs_preds, [sparse_t, preds], 1)

        class_change_heatmap(target_vs_preds, os.path.join(out_dir, "target_vs_preds.png"), class_names = class_names, percentages = False)
        class_change_heatmap(target_vs_preds, os.path.join(out_dir, "percent_target_vs_preds.png"), class_names = class_names, annotate = False)

        print("Total: %d" % len(data_x))
        print("Correct: %d" % correct)
        print("Correct / Total: %.3f" % (float(correct) / len(data_x)))

    print("Done!")