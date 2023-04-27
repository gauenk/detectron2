
import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image
from pathlib import Path

from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager

from .. import DatasetCatalog, MetadataCatalog

logger = logging.getLogger(__name__)
__all__ = ["register_youtube_instances"]

def register_youtube_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    base = Path(__file__).parents[3].resolve()
    image_root = str(base / image_root)
    json_file = str(base / json_file)
    # print(json_file, image_root, name)
    DatasetCatalog.register(name, lambda: load_youtube_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="youtube", **metadata
    )

def convert_to_coco_json(dataset_dicts, dataset_name, output_file, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    # TODO: The dataset or the conversion script *may* change,
    # a checksum would be useful for validating the cached data
    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.warning(
                f"Using previously cached COCO format annotations at '{output_file}'. "
                "You need to clear the cache file if your dataset has been modified."
            )
        else:
            logger.info(f"Converting annotations of dataset '{dataset_name}' to COCO format ...)")
            coco_dict = convert_to_coco_dict(dataset_dicts,dataset_name)
            logger.info(f"Caching COCO format annotations at '{output_file}' ...")
            tmp_file = output_file + ".tmp"
            with PathManager.open(tmp_file, "w") as f:
                json.dump(coco_dict, f)
            shutil.move(tmp_file, output_file)

def convert_to_coco_dict(dataset_dicts,dataset_name):
    """
    Convert an instance detection/segmentation or keypoint detection dataset
    in detectron2's standard format into COCO json format.

    Generic dataset description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

    COCO data format description can be found here:
    http://cocodataset.org/#format-data

    Args:
        dataset_name (str):
            name of the source dataset
            Must be registered in DatastCatalog and in detectron2's standard format.
            Must have corresponding metadata "thing_classes"
    Returns:
        coco_dict: serializable dict in COCO json format
    """

    # dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa
    # categories = metadata.thing_classes
    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    logger.info("Converting dataset dicts into COCO format")
    images = []
    annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        T = image_dict['video'].shape[0]
        for t in range(T):
            # print("[youtube] image_id: ","%d_%d" % (image_id,t),flush=True)
            coco_image = {
                "id": image_dict.get("image_id", image_id) + "_%d" % t,
                "width": int(image_dict["width"]),
                "height": int(image_dict["height"]),
                "file_name": str(image_dict["file_name"][t]),
            }
            images.append(coco_image)
            anns_per_image = image_dict.get("annotations", [])
            for annotation in anns_per_image[t]:
                # create a new dict with only COCO fields
                coco_annotation = {}

                # COCO requirement: XYWH box format for axis-align and XYWHA for rotated
                bbox = annotation["bbox"]
                if isinstance(bbox, np.ndarray):
                    if bbox.ndim != 1:
                        raise ValueError(f"bbox has to be 1-dimensional. Got shape={bbox.shape}.")
                    bbox = bbox.tolist()
                if len(bbox) not in [4, 5]:
                    raise ValueError(f"bbox has to has length 4 or 5. Got {bbox}.")
                from_bbox_mode = annotation["bbox_mode"]
                to_bbox_mode = BoxMode.XYWH_ABS if len(bbox) == 4 else BoxMode.XYWHA_ABS
                bbox = BoxMode.convert(bbox, from_bbox_mode, to_bbox_mode)

                # COCO requirement: instance area
                if "segmentation" in annotation:
                    # Computing areas for instances by counting the pixels
                    segmentation = annotation["segmentation"]
                    # TODO: check segmentation type: RLE, BinaryMask or Polygon
                    if isinstance(segmentation, list):
                        polygons = PolygonMasks([segmentation])
                        area = polygons.area()[0].item()
                    elif isinstance(segmentation, dict):  # RLE
                        area = mask_util.area(segmentation).item()
                    else:
                        raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
                else:
                    # Computing areas using bounding boxes
                    if to_bbox_mode == BoxMode.XYWH_ABS:
                        bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
                        area = Boxes([bbox_xy]).area()[0].item()
                    else:
                        area = RotatedBoxes([bbox]).area()[0].item()

                if "keypoints" in annotation:
                    keypoints = annotation["keypoints"]  # list[int]
                    for idx, v in enumerate(keypoints):
                        if idx % 3 != 2:
                            # COCO's segmentation coordinates are floating points in [0, H or W],
                            # but keypoint coordinates are integers in [0, H-1 or W-1]
                            # For COCO format consistency we substract 0.5
                            # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
                            keypoints[idx] = v - 0.5
                    if "num_keypoints" in annotation:
                        num_keypoints = annotation["num_keypoints"]
                    else:
                        num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

                # COCO requirement:
                #   linking annotations to images
                #   "id" field must start with 1
                coco_annotation["id"] = len(annotations) + 1
                coco_annotation["image_id"] = coco_image["id"]
                coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
                coco_annotation["area"] = float(area)
                coco_annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
                coco_annotation["category_id"] = int(reverse_id_mapper(annotation["category_id"]))

                # Add optional fields
                if "keypoints" in annotation:
                    coco_annotation["keypoints"] = keypoints
                    coco_annotation["num_keypoints"] = num_keypoints

                if "segmentation" in annotation:
                    segm = annotation["segmentation"]
                    segm = [s.tolist() for s in segm]
                    # seg = coco_annotation["segmentation"] = annotation["segmentation"]
                    seg = coco_annotation["segmentation"] = segm
                    if isinstance(seg, dict):  # RLE
                        counts = seg["counts"]
                        if not isinstance(counts, str):
                            # make it json-serializable
                            seg["counts"] = counts.decode("ascii")

                annotations.append(coco_annotation)

    logger.info(
        "Conversion finished, "
        f"#images: {len(images)}, #annotations: {len(annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {"info": info, "images": images, "categories": categories, "licenses": None}
    # print(coco_dict)
    if len(annotations) > 0:
        coco_dict["annotations"] = annotations
    return coco_dict

