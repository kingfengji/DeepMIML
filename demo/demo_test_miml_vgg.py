from __future__ import print_function
import argparse
import numpy as np
import sys
sys.path.insert(0, "lib")

from cocodemo import COCODataset, COCODataLayer
from deepmiml.utils import load_keras_model, evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, required=True,
            help="model path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model
    batch_size = 128

    model = load_keras_model(model_path)
    print("Compiling Deep MIML Model...")
    model.compile(optimizer="adadelta", loss="binary_crossentropy")

    # crate data layer
    dataset = COCODataset("data/coco", "val", "2014")
    data_layer = COCODataLayer(dataset, batch_size=batch_size)

    print("Start Predicting...")
    num_images = dataset.num_images
    y_pred = np.zeros((num_images, dataset.num_classes))
    y_gt = np.zeros((num_images, dataset.num_classes))
    for i in range(0, num_images, batch_size):
        if i // batch_size % 10 == 0:
            print("[progress] ({}/{})".format(i, num_images))
        x_val_mini, y_val_mini = data_layer.get_data(i, i + batch_size)
        y_pred_mini = model.predict(x_val_mini)
        y_pred[i: i + batch_size] = y_pred_mini
        y_gt[i: i + batch_size] = y_val_mini
    evaluate(dataset.classes, y_gt, y_pred, threshold_value=0.5)
