# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf

'''
/home/yjr/softWares/anaconda/bin/python2.7 /home/yjr/PycharmProjects/Faster-RCNN_TF/tools/eval.py
++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--++--
/home/yjr/PycharmProjects/Faster-RCNN_TF
model restore from : /home/yjr/PycharmProjects/Faster-RCNN_TF/output/trained_weights/FasterRCNN_20180527/voc_70001model.ckpt

cls : sheep|| Recall: 0.884297520661 || Precison: 0.0012608705899|| AP: 0.740768233287
____________________
cls : horse|| Recall: 0.962643678161 || Precison: 0.00190436127154|| AP: 0.892663858093
____________________
cls : bicycle|| Recall: 0.905044510386 || Precison: 0.00158366694186|| AP: 0.802685210134
____________________
cls : bottle|| Recall: 0.748400852878 || Precison: 0.00187884400242|| AP: 0.557355972535
____________________
cls : cow|| Recall: 0.94262295082 || Precison: 0.00126077137282|| AP: 0.833810598769
____________________
cls : sofa|| Recall: 0.953974895397 || Precison: 0.00153218598587|| AP: 0.693466349735
____________________
cls : bus|| Recall: 0.957746478873 || Precison: 0.00115234055053|| AP: 0.851274156507
____________________
cls : dog|| Recall: 0.963190184049 || Precison: 0.00274060281625|| AP: 0.88796447487
____________________
cls : cat|| Recall: 0.972067039106 || Precison: 0.0018943517833|| AP: 0.914821297181
____________________
cls : person|| Recall: 0.880521201413 || Precison: 0.0228400224562|| AP: 0.811749593466
____________________
cls : train|| Recall: 0.925531914894 || Precison: 0.00151498441482|| AP: 0.817050652016
____________________
cls : diningtable|| Recall: 0.864077669903 || Precison: 0.00118420352334|| AP: 0.627352986434
____________________
cls : aeroplane|| Recall: 0.877192982456 || Precison: 0.00148813066978|| AP: 0.78654783719
____________________
cls : car|| Recall: 0.93089092423 || Precison: 0.00647080613048|| AP: 0.868649445795
____________________
cls : pottedplant|| Recall: 0.729166666667 || Precison: 0.00211401166934|| AP: 0.470034802497
____________________
cls : tvmonitor|| Recall: 0.905844155844 || Precison: 0.00167818539437|| AP: 0.767061578407
____________________
cls : chair|| Recall: 0.820105820106 || Precison: 0.00360741964764|| AP: 0.502862355683
____________________
cls : bird|| Recall: 0.893246187364 || Precison: 0.00230857155727|| AP: 0.790455299012
____________________
cls : boat|| Recall: 0.863117870722 || Precison: 0.00130025604161|| AP: 0.605100112067
____________________
cls : motorbike|| Recall: 0.907692307692 || Precison: 0.00167738487169|| AP: 0.809558599055
____________________
mAP is : 0.751561670637

Process finished with exit code 0

'''
# ------------------------------------------------
VERSION = 'FasterRCNN_20180527'
NET_NAME = 'resnet_v1_50' #'MobilenetV2'
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
DECAY_STEP = [50000, 70000]  # 50000, 70000
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



