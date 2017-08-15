from __future__ import print_function
import cv2
import os.path as osp
import numpy as np

from pycocotools.coco import COCO


class COCODataset(object):

    def __init__(self, data_dir, image_set, year):
        self.name = 'coco_' + year + '_' + image_set
        # name, paths
        self.year = year
        self.image_set = image_set
        self.data_dir = data_dir

        # COCO API
        self.COCO = COCO(self._get_ann_file())
        cats = self.COCO.loadCats(self.COCO.getCatIds())
        self.classes = tuple([c['name'] for c in cats])
        self.num_classes = len(self.classes)
        # e.g: 'person' -> 0, 'toothbrush' -> 79
        self.class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        # e.g: 'person' -> 1, 'toothbrush' -> 90
        self.class_to_coco_cat_id = dict(zip([c['name'] for c in cats], self.COCO.getCatIds()))
        # coco' cat_id (1,,90) -> label_id (0,,79)
        self.coco_cat_id_to_class_ind = dict([(self.class_to_coco_cat_id[cls],
                                          self.class_to_ind[cls])
                                         for cls in self.classes])

        self.image_index = self._load_image_set_index()
        self.num_images = len(self.image_index)

        coco_name = image_set + year  # e.g., "val2014"
        self.data_name = coco_name

    def _get_ann_file(self):
        """
        Get the coco's annotation file path
            When image_set is train|val, it's located in annotations/$image_set/instances_$year.json
            When image_set is test, it's located in annotations/test/image_info_$year.json
        Return:
            anno_path (str)
        """
        prefix = 'instances' if self.image_set.find('test') == -1 else 'image_info'
        return osp.join(self.data_dir, 'annotations',
                        prefix + '_' + self.image_set + self.year + '.json')

    def _load_image_set_index(self):
        """
        Load image ids.
        """
        image_ids = self.COCO.getImgIds()
        return image_ids

    def image_at(self, i):
        im_path = self.image_path_at(i)
        im = cv2.imread(im_path)
        return im

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self.image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   train2014/COCO_train2014_000000119993.jpg
        image = self.COCO.loadImgs(index)[0]
        image_path = osp.join(self.data_dir, self.data_name, image['file_name'])
        assert osp.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    def gt_bag_labels(self):
        if hasattr(self, '_gt_bag_labels'):
            return self._gt_bag_labels
        self._gt_bag_labels = np.asarray([self._load_coco_annotation(index)
                    for index in self.image_index], dtype=np.int16)
        return self._gt_bag_labels

    def _load_coco_annotation(self, index):
        annIds = self.COCO.getAnnIds(imgIds=index, iscrowd=None)
        objs = self.COCO.loadAnns(annIds)
        bag_labels = np.zeros(self.num_classes)
        for obj in objs:
            cls = self.coco_cat_id_to_class_ind[obj['category_id']]
            bag_labels[cls] = 1
        return bag_labels


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    coco = COCODataset('data/coco', 'train', '2014')
    idx = 1
    print('Test index: ', idx)
    print('image path: ', coco.image_path_at(idx))
    gt_bag_labels = coco.gt_bag_labels()
    print('labels: ', [coco.classes[i] for i in range(80) if gt_bag_labels[idx, i]])
    image = coco.image_at(idx)
    plt.imshow(image[:, :, (2, 1, 0)])
    plt.show()
