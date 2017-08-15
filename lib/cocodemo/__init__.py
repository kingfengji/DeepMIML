import os.path as osp

from cocodemo.coco_dataset import COCODataset
from cocodemo.coco_data_layer import COCODataLayer


def get_coco_cls_names():
    coco_classes_path = osp.abspath(osp.join(__file__, osp.pardir, "coco_classes.txt"))
    cls_names = []
    with open(coco_classes_path, "r") as f:
        cls_names = [row.strip() for row in f.readlines() if len(row.strip()) > 0]
    return cls_names
