from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import numpy as np
import cv2
import random
import notebook_utils as nutils
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data.detection_utils import read_image
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

templates = dict(
      filename = '',
      folder = '',
      path = '',
      source= dict(database = 'Unknown'),
      size = dict(width = '1000', height='1000', depth='1'),
      segmented = '0',
      object = []
    )

def get_predictor(checkpoint_path, classes, input_min_size, min_score=0.7):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min_score
    cfg.MODEL.WEIGHTS = checkpoint_path
    cfg.INPUT.MIN_SIZE_TEST = input_min_size

    predictor = DefaultPredictor(cfg)
    
    return predictor
    
    
def infer_and_show(predictor, file, visual_scale=1):
    bboxes, classes, scores, v = infer(predictor, file, visualize=True, visual_scale=visual_scale)
    
def infer(name, predictor, file, visualize=False, visual_scale = 1, draw_box=False):
    metadata = MetadataCatalog.get("{}_train".format(name))
    
    im = read_image(str(file), format="BGR") if isinstance(file, str) else file
    
    outputs = predictor(im)
    instances = outputs['instances']
    
        
    bboxes = np.array(instances.get('pred_boxes').tensor.cpu())
    scores = np.array(instances.get('scores').cpu())
    classes = np.array(instances.get('pred_classes').cpu())
    
    v = None
    if visualize:
        v = Visualizer(im[:, :, ::-1],
                            metadata=metadata, 
                            scale=visual_scale, 
                            instance_mode=ColorMode.IMAGE_BW)
        
        for bbox in bboxes:
            v.draw_box(bbox, alpha = 1)

    
    return bboxes, classes, scores, v


# def infer_and_export_label(img, )