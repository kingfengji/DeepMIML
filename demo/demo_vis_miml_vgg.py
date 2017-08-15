from __future__ import print_function
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, "lib")

from cocodemo import COCODataset, COCODataLayer
from deepmiml.deepmiml import DeepMIML
from deepmiml.utils import load_keras_model
from deepmiml.vis_utils import plot_instance_attention, plot_instance_probs_heatmap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="data", type=int, default=0,
            help="image path")
    parser.add_argument("--model", dest="model", type=str, required=True,
            help="model path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    bag_id = args.data
    miml_model_path = args.model

    dataset = COCODataset("data/coco", "val", "2014")
    cls_names = dataset.classes
    data_layer = COCODataLayer(dataset)

    model = load_keras_model(miml_model_path)
    # model.summary()
    deepmiml = DeepMIML(model=model)

    im = dataset.image_at(bag_id)
    data_x, data_y = data_layer.get_data(bag_id, bag_id + 1)

    # shape = (n_instances, n_features)
    bag_features = deepmiml.get_bag_features(data_x)[0]
    print("bag_features.shape=", bag_features.shape)
    # shape = (n_instances, K, L)
    probs_cube = deepmiml.get_subconcept_cube_baglevel(data_x)[0]
    print("probs_cube.shape=", probs_cube.shape)
    # shape = (n_instances, K, L)
    instance_probs = deepmiml.get_predictions_instancelevel(data_x)[0]
    print("instance_probs.shape=", instance_probs.shape)

    # plot instance label score
    plot_instance_probs_heatmap(instance_probs)

    # plot instances attention
    # label_id_list = np.where(data_y[0] > 0)[0]
    label_id_list = np.where(deepmiml.get_predictions(data_x)[0] > 0.5)[0]
    label_name_list = [cls_names[i] for i in label_id_list]
    instance_points, instance_labels = [], []
    for _i, label_id in enumerate(label_id_list):
        max_instance_id = np.argmax(instance_probs[:, label_id])
        conv_y, conv_x = max_instance_id / 14, max_instance_id % 14
        instance_points.append(((conv_x * 16 + 8), (conv_y * 16 + 8)))
        instance_labels.append(label_name_list[_i])
    im_plot = cv2.resize(im, (224, 224)).astype(np.uint8)[:, :, (2, 1, 0)]
    plot_instance_attention(im_plot, instance_points, instance_labels)

    import IPython; IPython.embed()
