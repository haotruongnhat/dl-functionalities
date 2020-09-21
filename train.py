import torch, torchvision
from pascal_voc_tools import *
parser = XmlParser()

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import os
# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.utils.visualizer import ColorMode
from detectron2.data.detection_utils import read_image
from detectron2.structures import BoxMode
from pathlib import Path
from joblib import Parallel, delayed

from tqdm import tqdm

import notebook_utils as nutils

def check_file_verified(path):
  if isinstance(path, list):
    return Parallel(n_jobs=40)(delayed(check_file_verified)(p) for p in path)
  else:
    with open(path, "r") as f:
      content = f.read()
    
    return bool(content.count('verified'))

def calculate_proportion(sum_, train_ratio, val_ratio):
  lenghts = []
  lenghts.append(int(np.ceil(sum_*train_ratio)))

  # rest_ratio = (1-train_ratio)/2
  lenghts.append(int(sum_*val_ratio))

  lenghts.append(sum_ - sum(lenghts))

  return lenghts

def random_indices(output_size, total_num):
  from numpy.random import default_rng

  rng = default_rng()
  index = rng.choice(total_num, size=output_size, replace=False)
  
  return list(index)

def generate_dataset_indices(total_samples, test_ratio, val_ratio=0.0):
  train_ratio = 1.0 - test_ratio - val_ratio

  train_num_samples, val_num_samples, test_num_samples = calculate_proportion(total_samples, train_ratio, val_ratio)
  train_indices = random_indices(train_num_samples, total_samples)
  val_indices = random_indices(val_num_samples, total_samples - train_num_samples)
  
  return train_indices, val_indices

def del_numpy(l, id_to_del):
  return np.delete(l, id_to_del)

def output_set(paths, all_to_train=False):
  path_array = np.array(paths)

  # is_verified = np.array(check_file_verified(xml_paths))

  # unverified_annotations_indices = list(np.where(is_verified == False)[0])

  # unverified_to_test_list = list(path_array[unverified_annotations_indices])

  # path_the_rest = del_numpy(path_array, unverified_annotations_indices)

  path_the_rest = path_array
  subset_indices = generate_dataset_indices(len(path_the_rest), 0.15 if (not all_to_train) else 0)
  train_files = path_the_rest[subset_indices[0]]
  path_the_rest = del_numpy(path_the_rest, subset_indices[0])
  val_files = path_the_rest[subset_indices[1]]
  test_files = del_numpy(path_the_rest, subset_indices[1])

  # train_files = path_the_rest if not all_to_train else list(path_array)
  # val_files = []
  # test_files = []

  return {'train': list(train_files), 
          'val': list(val_files), 
          'test': list(test_files)}

def parse_file(path):
    annotation = parser.load(path)
    return annotation

def list_class(paths):
    classes = []
    for i, path in enumerate(paths):
        annotation = parser.load(path)
        
        for obj in annotation['object']:
            try:
                classes.index(obj['name'])
            except:
                print('Found class {} in file {}'.format(obj['name'], str(path)))
                classes.append(obj['name'])
                
    return classes

def get_dicts(paths, classes):
  # xml_paths = list(annotations_path.glob("*.xml"))

  dataset_dicts = []

  for i, path in enumerate(paths):
    annotation = parser.load(path)

    try: 

        record = {}
        record['file_name'] = str(path.parent / (path.stem + '.JPG'))
        record['image_id'] = i
        record['height'] = int(annotation['size']['height'])
        record['width'] = int(annotation['size']['width'])

        objs = []
        for obj in annotation['object']:
          objs.append({
              "bbox": list(map(float,[obj['bndbox']['xmin'],obj['bndbox']['ymin'],obj['bndbox']['xmax'],obj['bndbox']['ymax']])),
              "bbox_mode": BoxMode.XYXY_ABS,
              "category_id": classes.index(obj['name']) if (len(classes) > 1) else 0,
              "iscrowd": 0
          })
        record["annotations"] = objs
        dataset_dicts.append(record)
        
    except:
        continue

  return dataset_dicts

def add_catalog(dataset, name, classes):
    DatasetCatalog.clear()
    for d in ['train', 'val', 'test']:
      DatasetCatalog.register("{}_{}".format(name, d), lambda d=d: get_dicts(dataset[d], classes))
      MetadataCatalog.get("{}_{}".format(name, d)).set(thing_classes=classes)

def get_catalog(name, set_type="train"):
    return MetadataCatalog.get("{}_{}".format(name, set_type))
    
def get_dect_cfg(name, classes, min_size=None, num_iter=None):
    num_classes = len(classes)

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("{}_train".format(name),)
    cfg.DATASETS.TEST = ("{}_test".format(name),)
    # cfg.DATALOADER.NUM_WORKERS = 0#40
    cfg.SOLVER.IMS_PER_BATCH = 4#4
    cfg.SOLVER.BASE_LR = 0.00025#0.00025  # pick a good LR
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128#128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    # cfg.MODEL.WEIGHTS = os.path.join("Checkpoints/steel-faster-rcnn_detectron2_v3.pth")
#     MIN_SIZE_TRAIN: (640, 672, 704, 736, 768, 800)
    if num_iter:
        cfg.SOLVER.MAX_ITER = num_iter    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset

    if min_size:
        cfg.INPUT.MIN_SIZE_TRAIN = min_size
        
    return cfg
    
    
def train(dataset, name, classes, min_size=None, num_iter=3000):
    print("Number of samples: Train = {} - Val = {} - Test = {}".format(len(dataset['train']),len(dataset['val']),len(dataset['test'])))
    
    add_catalog(dataset, name, classes)
    cfg = get_dect_cfg(name, classes, min_size, num_iter)
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    torch.save(trainer.model.state_dict(), os.path.join("{}.pth".format(name)))
