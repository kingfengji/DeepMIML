from __future__ import print_function
from keras.models import model_from_json
import numpy as np
import os
import os.path as osp
from sklearn.metrics import average_precision_score, f1_score, hamming_loss


def save_keras_model(model, output_dir):
    print("Saving model in ", output_dir)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    model_json_path = osp.join(output_dir, "arch.json")
    model_h5_path = osp.join(output_dir, "weights.h5")
    with open(model_json_path, "w") as f:
        f.write(model.to_json())
    model.save_weights(model_h5_path, overwrite=True)


def load_keras_model(output_dir):
    print("Loading model from ", output_dir)
    model_json_path = osp.join(output_dir, "arch.json")
    model_h5_path = osp.join(output_dir, "weights.h5")
    with open(model_json_path) as f:
        model = model_from_json(f.read())
    model.load_weights(model_h5_path)
    return model


def evaluate(classes, y_gt, y_pred, threshold_value=0.5):
    """
    Arguments:
        y_gt (num_bag x L): groud truth
        y_pred (num_bag x L): prediction
    """
    print("thresh = {:.6f}".format(threshold_value))

    y_pred_bin = y_pred >= threshold_value

    score_f1_macro = f1_score(y_gt, y_pred_bin, average="macro")
    print("Macro f1_socre = {:.6f}".format(score_f1_macro))

    score_f1_micro = f1_score(y_gt, y_pred_bin, average="micro")
    print("Micro f1_socre = {:.6f}".format(score_f1_micro))

    # hamming loss
    h_loss = hamming_loss(y_gt, y_pred_bin)
    print("Hamming Loss = {:.6f}".format(h_loss))

    mAP = average_precision_score(y_gt, y_pred)
    print("mAP = {:.2f}%".format(mAP * 100))
    # ap_classes = []
    # for i, cls in enumerate(classes):
    #     ap_cls = average_precision_score(y_gt[:, i], y_pred[:, i])
    #     ap_classes.append(ap_cls)
    #     print("AP({}) = {:.2f}%".format(cls, ap_cls * 100))
    # print("mAP = {:.2f}%".format(np.mean(ap_classes) * 100))
