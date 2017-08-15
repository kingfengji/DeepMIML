from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.convolutional import Convolution2D
import sys
sys.path.insert(0, "lib")

from cocodemo import COCODataset, COCODataLayer
from deepmiml.deepmiml import DeepMIML
from deepmiml.utils import save_keras_model
from cocodemo.vgg_16 import VGG_16


if __name__ == "__main__":
    loss = "binary_crossentropy"
    nb_epoch = 10
    batch_size = 32
    L = 80
    K = 20
    model_name = "miml_vgg_16"

    # crate data layer
    dataset = COCODataset("data/coco", "train", "2014")
    data_layer = COCODataLayer(dataset, batch_size=batch_size)

    vgg_model_path = "models/imagenet/vgg/vgg16_weights.h5"
    base_model = VGG_16(vgg_model_path)
    base_model = Sequential(layers=base_model.layers[: -7])
    base_model.add(Convolution2D(512, 1, 1, activation="relu"))
    base_model.add(Dropout(0.5))

    deepmiml = DeepMIML(L=L, K=K, base_model=base_model)
    # deepmiml.model.summary()

    print("Compiling Deep MIML Model...")
    deepmiml.model.compile(optimizer="adadelta", loss=loss, metrics=["accuracy"])

    print("Start Training...")
    samples_per_epoch = data_layer.num_images
    deepmiml.model.fit_generator(data_layer.generate(),
            samples_per_epoch=samples_per_epoch,
            nb_epoch=nb_epoch)

    save_keras_model(deepmiml.model, "outputs/{}/{}".format(dataset.name, model_name))
