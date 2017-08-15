from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Reshape, Permute, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D


MIML_FIRST_LAYER_NAME = "miml/first_layer"
MIML_CUBE_LAYER_NAME = "miml/cube"
MIML_TABLE_LAYER_NAME = "miml/table"
MIML_OUTPUT_LAYER_NAME = "miml/output"


def create_miml_model(base_model, L, K, name="miml"):
    """
    Arguments:
        base_model (Sequential):
            A Neural Network in keras form (e.g. VGG, GoogLeNet)
        L (int):
            number of labels
        K (int):
            number of sub categories
    """
    model = Sequential(layers=base_model.layers, name=name)

    # input: feature_map.shape = (n_bags, C, H, W)
    _, C, H, W = model.layers[-1].output_shape
    print("Creating miml... input feature_map.shape={},{},{}".format(C, H, W))
    n_instances = H * W

    # shape -> (n_bags, (L * K), n_instances, 1)
    model.add(Convolution2D(L * K, 1, 1, name=MIML_FIRST_LAYER_NAME))
    # shape -> (n_bags, L, K, n_instances)
    model.add(Reshape((L, K, n_instances), name=MIML_CUBE_LAYER_NAME))
    # shape -> (n_bags, L, 1, n_instances)
    model.add(MaxPooling2D((K, 1), strides=(1, 1)))
    # softmax
    model.add(Reshape((L, n_instances)))
    model.add(Permute((2, 1)))
    model.add(Activation("softmax"))
    model.add(Permute((2, 1)))
    model.add(Reshape((L, 1, n_instances), name=MIML_TABLE_LAYER_NAME))
    # shape -> (n_bags, L, 1, 1)
    model.add(MaxPooling2D((1, n_instances), strides=(1, 1)))
    # shape -> (n_bags, L)
    model.add(Reshape((L,), name=MIML_OUTPUT_LAYER_NAME))
    return model


class DeepMIML(object):
    def __init__(self, L=None, K=None, base_model=None, model=None):
        """
        When model is None:
            The DeepMIML model will be created by appending the required MIML layers after base_model
        Otherwise:
            model will be used as the DeepMIML model

        Arguments:
            L (int): Number of classes
            K (int): Number of subconcepts
            base_model (keras.Model):
                To which model the MIML layers will be append
            model (keras.Model):
                The DeepMIML Model you already have
        """
        if model is None:
            self.model = create_miml_model(base_model, L, K)
        else:
            self.model = model

        self.name2layer = {}
        for layer in self.model.layers:
            self.name2layer[layer.name] = layer

    def _get_layer_input(self, x, layer_name):
        """
        Get the specified layer's input

        Argumetns:
            x (ndarray): shape = (n_bags, in_features, in_height, in_width)
        Retrun:
            input_ (ndarray):
                The layer's input
        """
        get_input = K.function([self.model.layers[0].input, K.learning_phase()],
                [self.name2layer[layer_name].get_input_at(0)])
        # 0 represent test phase
        input_ = get_input([x, 0])[0]
        return input_

    def _get_layer_output(self, x, layer_name):
        """
        Get the specified layer's output

        Argumetns:
            x (ndarray): shape = (n_bags, in_features, in_height, in_width)
        Retrun:
            output (ndarray):
                The layer's output
        """
        get_output = K.function([self.model.layers[0].input, K.learning_phase()],
                [self.name2layer[layer_name].get_output_at(0)])
        # 0 represent test phase
        output = get_output([x, 0])[0]
        return output

    def get_bag_features(self, x):
        """
        Get the input features for MIML

        Argumetns:
            x (ndarray): shape = (n_bags, in_features, in_height, in_width)
        Return:
            bag_features (ndarray): shape = (n_bags, n_instances, n_input_features)
        """
        # shape = (n_bags, n_input_features, input_height, input_width)
        bag_features = self._get_layer_input(x, MIML_FIRST_LAYER_NAME)
        bag_features = bag_features.reshape(bag_features.shape[0], bag_features.shape[1], -1).transpose((0, 2, 1))
        return bag_features

    def get_subconcept_cube_baglevel(self, x):
        """
        Get the subconcept cube of bag

        Argumetns:
            x (ndarray): shape = (n_bags, in_features, in_height, in_width)
        Return:
            cube (ndarray): shape = (n_bags, n_instances, K, L)
                cube[i] represent the subconcepts of instance $i
        """
        # shape = (n_bags, L, K, n_instances)
        cube = self._get_layer_output(x, MIML_CUBE_LAYER_NAME)
        cube = cube.transpose((0, 3, 2, 1))
        return cube

    def get_subconcept_table_instancelevel(self, x, instance_id):
        """
        Get the subconcept table of a particular instance

        Argumetns:
            x (ndarray): shape = (n_bags, in_features, in_height, in_width)
            instance_id (int):
                the instance id
        Return:
            table (ndarray): shape = (K, L)
                the subconcepts table of instance $i
        """
        cube = self.get_subconcept_cube_baglevel(x)
        table = cube[:, instance_id, :, :]
        return table

    def get_predictions(self, x):
        """
        Argumetns:
            x (ndarray): shape = (n_bags, in_features, in_height, in_width)
        Return:
            probs (ndarray): shape = (L, )
                The classes probability distribution
        """
        # shape = (n_bags, L)
        probs = self._get_layer_output(x, MIML_OUTPUT_LAYER_NAME)
        return probs

    def get_predictions_instancelevel(self, x):
        """
        Argumetns:
            x (ndarray): shape = (n_bags, in_features, in_height, in_width)
        Return:
            probs (ndarray): shape = (n_bags, n_instances, L)
        """
        # shape = (n_bags, L, 1, n_instances)
        probs = self._get_layer_output(x, MIML_TABLE_LAYER_NAME)
        probs = probs.transpose((0, 3, 1, 2))[:, :, :, 0]
        return probs
