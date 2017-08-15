import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import os.path as osp


def plot_instance_attention(im, instance_points, instance_labels, save_path=None):
    """
    Arguments:
        im (ndarray): shape = (3, im_width, im_height)
            the image array
        instance_points: List of (x, y) pairs
            the instance's center points
        instance_labels: List of str
            the label name of each instance
    """
    fig, ax = plt.subplots()
    ax.imshow(im)
    for i, (x_center, y_center) in enumerate(instance_points):
        label = instance_labels[i]
        center = plt.Circle((x_center, y_center), 10, color="r", alpha=0.5)
        ax.add_artist(center)
        ax.text(x_center, y_center, str(label), fontsize=18,
                bbox=dict(facecolor="blue", alpha=0.7), color="white")
    if save_path:
        if not osp.exists(osp.dirname(save_path)):
            os.makedirs(osp.dirname(save_path))
        fig.savefig(save_path)
    else:
        plt.show()


def plot_instance_probs_heatmap(instance_probs, save_path=None):
    """
    Arguments:
        instance_probs (ndarray): shape = (n_instances, n_labels)
            the probability distribution of each instance
    """
    n_instances, n_labels = instance_probs.shape
    fig, ax = plt.subplots()
    ax.set_title("Instance-Label Scoring Layer Visualized")

    cax = ax.imshow(instance_probs, vmin=0, vmax=1, cmap=cm.hot, aspect=float(n_labels) / n_instances)
    cbar_ticks = list(np.linspace(0, 1, 11))
    cbar = fig.colorbar(cax, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(map(str, cbar_ticks))

    if save_path:
        if not osp.exists(osp.dirname(save_path)):
            os.makedirs(osp.dirname(save_path))
        fig.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    instance_probs = np.random.random((196, 80)) * 0.5
    plot_instance_probs_heatmap(instance_probs)
