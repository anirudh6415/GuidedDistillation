# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.utils.file_io import PathManager


TB_CATEGORIES =[
    # { "id": 1, "name": "ActiveTuberculosis", "supercategory": "Tuberculosis" },
    # {
    #  "id": 2,
    #  "name": "ObsoletePulmonaryTuberculosis",
    #  "supercategory": "Tuberculosis"
    # },
    # {
    #  "id": 3,
    #  "name": "PulmonaryTuberculosis",
    #  "supercategory": "Tuberculosis"
    # }
    {
      "id": 1,
      "name": "TB",
      "supercategory": "Tuberculosis"
    }
    
  ]

_PREDEFINED_SPLITS = {
    # point annotations without masks
    "TB_instance_train": (
        "TBX11K/imgs",
        "TBX11K/annotations/json/all_train_2.json",
    ),
    "TB_instance_val": (
        "TBX11K/imgs",
        "TBX11K/annotations/json/all_test.json",
    ),
}


def _get_ade_instances_meta():
    thing_ids = [k["id"] for k in TB_CATEGORIES]
    assert len(thing_ids) == 1, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in TB_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def register_all_ade20k_instance(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_ade_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "C:/Users/aniru/Downloads/")
register_all_ade20k_instance(_root)