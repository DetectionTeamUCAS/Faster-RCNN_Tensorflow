# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf

'''
cls : sheep|| Recall: 0.904958677686 || Precison: 0.00100514505758|| AP: 0.796842233528
GLib-GIO-Message: 18:08:55.441: Using the 'memory' GSettings backend.  Your settings will not be saved or shared with other applications.
____________________
cls : horse|| Recall: 0.954022988506 || Precison: 0.00154345380331|| AP: 0.89332778065
____________________
cls : bicycle|| Recall: 0.916913946588 || Precison: 0.00131796700391|| AP: 0.838940379355
____________________
cls : bottle|| Recall: 0.754797441365 || Precison: 0.00155802315909|| AP: 0.59409144003
____________________
cls : cow|| Recall: 0.94262295082 || Precison: 0.00104145441373|| AP: 0.856768981614
____________________
cls : sofa|| Recall: 0.958158995816 || Precison: 0.0011761684643|| AP: 0.765890977118
____________________
cls : bus|| Recall: 0.943661971831 || Precison: 0.000950628786553|| AP: 0.842320499598
____________________
cls : dog|| Recall: 0.965235173824 || Precison: 0.00214377850045|| AP: 0.88499630618
____________________
cls : cat|| Recall: 0.932960893855 || Precison: 0.00149788547006|| AP: 0.884974245992
____________________
cls : person|| Recall: 0.877650176678 || Precison: 0.0167097655839|| AP: 0.815442624142
____________________
cls : train|| Recall: 0.90780141844 || Precison: 0.00116778731673|| AP: 0.7916043079
____________________
cls : diningtable|| Recall: 0.888349514563 || Precison: 0.00100377925633|| AP: 0.726617005258
____________________
cls : aeroplane|| Recall: 0.870175438596 || Precison: 0.00111434631008|| AP: 0.802605776982
____________________
cls : car|| Recall: 0.9367194005 || Precison: 0.00492263800889|| AP: 0.884210759442
____________________
cls : pottedplant|| Recall: 0.69375 || Precison: 0.00151877257635|| AP: 0.475030352567
____________________
cls : tvmonitor|| Recall: 0.912337662338 || Precison: 0.00127649522564|| AP: 0.79810054976
____________________
cls : chair|| Recall: 0.804232804233 || Precison: 0.00275795747846|| AP: 0.528492403027
____________________
cls : bird|| Recall: 0.886710239651 || Precison: 0.00183182331682|| AP: 0.807009364722
____________________
cls : boat|| Recall: 0.851711026616 || Precison: 0.00101167033999|| AP: 0.599356235684
____________________
cls : motorbike|| Recall: 0.904615384615 || Precison: 0.00127213801399|| AP: 0.818659097267
____________________
mAP is : 0.770264066041
This version is for Res101
'''

# ------------------------------------------------
VERSION = 'FasterRCNN_20180517'
NET_NAME = 'resnet_v1_101' #'MobilenetV2'
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
# INFERENCE_IMAGE_PATH = ROOT_PATH + '/tools/inference_image'
# INFERENCE_SAVE_PATH = ROOT_PATH + '/tools/inference_results'

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
