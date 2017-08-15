import cv2
import numpy as np


def im_to_blob(im):
    im = cv2.resize(im, (224, 224)).astype(np.float32)
    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68
    blob = im.transpose((2, 0, 1))
    return blob


class COCODataLayer(object):
    def __init__(self, dataset, batch_size=32):
        """
        Attibutes:
            dataset (COCODataset):
                dataset from which to load the data.
            batch_size (int):
                how many samples per batch to load
        """
        self.dataset = dataset
        self.batch_size = batch_size

        self.num_images = self.dataset.num_images
        self.data_y = self.dataset.gt_bag_labels()
        self.cur = 0
        self.perm = np.random.permutation(np.arange(self.num_images))

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self.perm = np.random.permutation(np.arange(self.num_images))
        self.cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self.cur + self.batch_size >= self.num_images:
            self._shuffle_roidb_inds()
        db_inds = self.perm[self.cur:self.cur + self.batch_size]
        self.cur += self.batch_size
        return db_inds

    def _get_next_minibatch(self):
        """
        Retrun:
            data_x: num_bag x 3(c) x 224(h) x 224(w)
            data_y: num_bag x 80
        """
        data_x = np.zeros((self.batch_size, 3, 224, 224), dtype=np.float32)
        data_y = np.zeros((self.batch_size, self.dataset.num_classes), dtype=np.float32)
        db_inds = self._get_next_minibatch_inds()
        for i, db_ind in enumerate(db_inds):
            im = self.dataset.image_at(db_ind)
            blob = im_to_blob(im)
            data_x[i] = blob
            data_y[i] = self.data_y[db_ind]
        return data_x, data_y

    def get_data(self, start, end):
        end = min(end, self.num_images)
        num_bag = end - start
        data_y = self.dataset.gt_bag_labels()[start: end]
        data_x = np.zeros((num_bag, 3, 224, 224))
        for i in range(start, end):
            im = self.dataset.image_at(i)
            blob = im_to_blob(im)
            data_x[i - start] = blob
        return data_x, data_y

    def generate(self):
        """
        This function is used for keras.Model.fit_generator
        """
        self.cur = 0
        while 1:
            x, y = self._get_next_minibatch()
            yield (x, y)


if __name__ == '__main__':
    from cocodemo.coco_dataset import COCODataset
    coco = COCODataset('data/coco', 'train', '2014')
    data_layer = COCODataLayer(coco)
    (x_train, y_train) = data_layer.generate().next()
    print(x_train.shape)
    print(y_train.shape)
