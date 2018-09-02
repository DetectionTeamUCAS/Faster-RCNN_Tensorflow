# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf

'''
cls : sheep|| Recall: 0.867768595041 || Precison: 0.00131288566016|| AP: 0.466752899093
____________________
cls : horse|| Recall: 0.936781609195 || Precison: 0.00217664182891|| AP: 0.704454703051
____________________
cls : bicycle|| Recall: 0.86646884273 || Precison: 0.00182144817606|| AP: 0.674277143751
____________________
cls : bottle|| Recall: 0.616204690832 || Precison: 0.00178924102748|| AP: 0.256914272734
____________________
cls : cow|| Recall: 0.926229508197 || Precison: 0.00140717910401|| AP: 0.535994662051
____________________
cls : sofa|| Recall: 0.92050209205 || Precison: 0.0015696347032|| AP: 0.462546085844
____________________
cls : bus|| Recall: 0.892018779343 || Precison: 0.00133138064173|| AP: 0.589536841927
____________________
cls : dog|| Recall: 0.907975460123 || Precison: 0.00290029264214|| AP: 0.376209487428
____________________
cls : cat|| Recall: 0.896648044693 || Precison: 0.00201439571266|| AP: 0.439673465315
____________________
cls : person|| Recall: 0.822659010601 || Precison: 0.0230289391851|| AP: 0.676651762397
____________________
cls : train|| Recall: 0.890070921986 || Precison: 0.00181387214731|| AP: 0.613459453384
____________________
cls : diningtable|| Recall: 0.849514563107 || Precison: 0.00120333633594|| AP: 0.521391389694
____________________
cls : aeroplane|| Recall: 0.80701754386 || Precison: 0.00149129859688|| AP: 0.565380801848
____________________
cls : car|| Recall: 0.892589508743 || Precison: 0.00688959298701|| AP: 0.750196152582
____________________
cls : pottedplant|| Recall: 0.639583333333 || Precison: 0.00199288533444|| AP: 0.244656318345
____________________
cls : tvmonitor|| Recall: 0.857142857143 || Precison: 0.00168739693456|| AP: 0.498885491685
____________________
cls : chair|| Recall: 0.727513227513 || Precison: 0.00353261567711|| AP: 0.277644881621
____________________
cls : bird|| Recall: 0.777777777778 || Precison: 0.0022371084277|| AP: 0.380395864449
____________________
cls : boat|| Recall: 0.77566539924 || Precison: 0.0013892483077|| AP: 0.381953563835
____________________
cls : motorbike|| Recall: 0.910769230769 || Precison: 0.00197399133044|| AP: 0.654647270689
____________________
mAP is : 0.503581125586

'''
# ------------------------------------------------
VERSION = 'FasterRCNN_20180516_mobile'
NET_NAME = 'MobilenetV2' #'MobilenetV2'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print (20*"++--")
print (ROOT_PATH)
GPU_GROUP = "0"
SHOW_TRAIN_INFO_INTE = 10
SMRY_ITER = 100
SAVE_WEIGHTS_INTE = 10000

SUMMARY_PATH = ROOT_PATH + '/output/summary'
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'
INFERENCE_IMAGE_PATH = ROOT_PATH + '/tools/inference_image'
INFERENCE_SAVE_PATH = ROOT_PATH + '/tools/inference_results'

if NET_NAME.startswith("resnet"):
    weights_name = NET_NAME
elif NET_NAME.startswith("MobilenetV2"):
    weights_name = "mobilenet/mobilenet_v2_1.0_224"
else:
    raise Exception('net name must in [resnet_v1_101, resnet_v1_50, MobilenetV2]')

PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
EVALUATE_DIR = ROOT_PATH + '/output/evaluate_result_pickle/'

# ------------------------------------------ Train config
RESTORE_FROM_RPN = False
IS_FILTER_OUTSIDE_BOXES = True
FIXED_BLOCKS = 1  # allow 0~3

RPN_LOCATION_LOSS_WEIGHT = 1.
RPN_CLASSIFICATION_LOSS_WEIGHT = 1.0

FAST_RCNN_LOCATION_LOSS_WEIGHT = 1.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 1.0
RPN_SIGMA = 3.0
FASTRCNN_SIGMA = 1.0


MUTILPY_BIAS_GRADIENT = None   # 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = None   #10.0  if None, will not clip

EPSILON = 1e-5
MOMENTUM = 0.9
LR = 0.001 # 0.001  # 0.0003
DECAY_STEP = [50000, 100000]  # 50000, 70000
MAX_ITERATION = 200000

# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'pascal'  # 'ship', 'spacenet', 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 600
IMG_MAX_LENGTH = 1000
CLASS_NUM = 20

# --------------------------------------------- Network_config
BATCH_SIZE = 1
INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01)
BBOX_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.001)
WEIGHT_DECAY = 0.00004 if NET_NAME.startswith('Mobilenet') else 0.0001


# ---------------------------------------------Anchor config
BASE_ANCHOR_SIZE_LIST = [256]  # can be modified
ANCHOR_STRIDE = [16]  # can not be modified in most situations
ANCHOR_SCALES = [0.5, 1., 2.0]  # [4, 8, 16, 32]
ANCHOR_RATIOS = [0.5, 1., 2.0]
ROI_SCALE_FACTORS = [10., 10., 5.0, 5.0]
ANCHOR_SCALE_FACTORS = None


# --------------------------------------------RPN config
KERNEL_SIZE = 3
RPN_IOU_POSITIVE_THRESHOLD = 0.7
RPN_IOU_NEGATIVE_THRESHOLD = 0.3
TRAIN_RPN_CLOOBER_POSITIVES = False

RPN_MINIBATCH_SIZE = 256
RPN_POSITIVE_RATE = 0.5
RPN_NMS_IOU_THRESHOLD = 0.7
RPN_TOP_K_NMS_TRAIN = 12000
RPN_MAXIMUM_PROPOSAL_TARIN = 2000

RPN_TOP_K_NMS_TEST = 6000  # 5000
RPN_MAXIMUM_PROPOSAL_TEST = 300  # 300


# -------------------------------------------Fast-RCNN config
ROI_SIZE = 14
ROI_POOL_KERNEL_SIZE = 2
USE_DROPOUT = False
KEEP_PROB = 1.0
SHOW_SCORE_THRSHOLD = 0.5  # only show in tensorboard

FAST_RCNN_NMS_IOU_THRESHOLD = 0.3  # 0.6
FAST_RCNN_NMS_MAX_BOXES_PER_CLASS = 100
FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.5
FAST_RCNN_IOU_NEGATIVE_THRESHOLD = 0.0   # 0.1 < IOU < 0.5 is negative
FAST_RCNN_MINIBATCH_SIZE = 256  # if is -1, that is train with OHEM
FAST_RCNN_POSITIVE_RATE = 0.25

ADD_GTBOXES_TO_TRAIN = False



