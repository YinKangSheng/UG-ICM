import os
import cv2
import torch
import random
import detectron2
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_pascal_voc
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
# Initialize the logger
# from detectron2.model_zoo import get_config_file,
from detectron2 import model_zoo
from PIL import ImageFile
import numpy as np
# ImageFile.LOAD_TRUNCATED_IMAGES = True

setup_logger()

dataset_name = "coco_8k_hy"  

json_file = "Coco/coco_2017.json"

image_dir = "path/to/images"

register_coco_instances(dataset_name, {}, json_file, image_dir)

# Get dataset metadata and dataset dictionaries
metadata = MetadataCatalog.get(dataset_name)
dataset_dicts = DatasetCatalog.get(dataset_name)

# Configure and load the pre-trained mask_rcnn model
cfg = get_cfg()

print("===============load model==============")
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
print("==========complete==================")

cfg.MODEL.DEVICE = "cuda"  #
cfg.DATASETS.TEST = (dataset_name, )  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)  

predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir="./mask/")
val_loader = build_detection_test_loader(cfg, dataset_name)

output_dir = "./mask/"
os.makedirs(output_dir, exist_ok=True)

# Perform inference and evaluation
print("Running inference...")
results = inference_on_dataset(predictor.model, val_loader, evaluator)
