# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
'''
cls : sheep|| Recall: 0.863636363636 || Precison: 0.00115267099792|| AP: 0.726904918741
____________________
cls : horse|| Recall: 0.951149425287 || Precison: 0.00165482624324|| AP: 0.857229276518
____________________
cls : bicycle|| Recall: 0.899109792285 || Precison: 0.00142927899243|| AP: 0.829476846459
____________________
cls : bottle|| Recall: 0.754797441365 || Precison: 0.00175056003086|| AP: 0.571565169065
____________________
cls : cow|| Recall: 0.94262295082 || Precison: 0.00118603164126|| AP: 0.795070234289
____________________
cls : sofa|| Recall: 0.928870292887 || Precison: 0.00129156121826|| AP: 0.706209140011
____________________
cls : bus|| Recall: 0.924882629108 || Precison: 0.00101847208508|| AP: 0.805255396726
____________________
cls : dog|| Recall: 0.969325153374 || Precison: 0.00243055733603|| AP: 0.843633210562
____________________
cls : cat|| Recall: 0.958100558659 || Precison: 0.00170786114043|| AP: 0.859281417826
____________________
cls : person|| Recall: 0.875883392226 || Precison: 0.0201840278485|| AP: 0.810128892526
____________________
cls : train|| Recall: 0.943262411348 || Precison: 0.00137269776395|| AP: 0.817213047381
____________________
cls : diningtable|| Recall: 0.893203883495 || Precison: 0.000994653736168|| AP: 0.689490640904
____________________
cls : aeroplane|| Recall: 0.877192982456 || Precison: 0.00130819505712|| AP: 0.783346011502
____________________
cls : car|| Recall: 0.926727726894 || Precison: 0.00557068209574|| AP: 0.864413714132
____________________
cls : pottedplant|| Recall: 0.725 || Precison: 0.00182820158549|| AP: 0.435668951097
____________________
cls : tvmonitor|| Recall: 0.886363636364 || Precison: 0.00150629831328|| AP: 0.730090265551
____________________
cls : chair|| Recall: 0.804232804233 || Precison: 0.00312483938942|| AP: 0.473597967821
____________________
cls : bird|| Recall: 0.871459694989 || Precison: 0.00199395830633|| AP: 0.733078865573
____________________
cls : boat|| Recall: 0.828897338403 || Precison: 0.00112203407278|| AP: 0.543688273498
____________________
cls : motorbike|| Recall: 0.901538461538 || Precison: 0.00140155845647|| AP: 0.789669117798
____________________
mAP is : 0.733250567899
'''

# ------------------------------------------------
VERSION = 'FasterRCNN_Res50_20180619'
NET_NAME = 'resnet_v1_50' #'MobilenetV2'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROO_PATH = os.path.abspath('../')
print (20*"++--")
print (ROO_PATH)
GPU_GROUP = "0"
SHOW_TRAIN_INFO_INTE = 10
SMRY_ITER = 100
SAVE_WEIGHTS_INTE = 10000

SUMMARY_PATH = ROO_PATH + '/output/summary'
TEST_SAVE_PATH = ROO_PATH + '/tools/test_result'
INFERENCE_IMAGE_PATH = ROO_PATH + '/tools/inference_image'
INFERENCE_SAVE_PATH = ROO_PATH + '/tools/inference_results'

if NET_NAME.startswith("resnet"):
    weights_name = NET_NAME
elif NET_NAME.startswith("MobilenetV2"):
    weights_name = "mobilenet/mobilenet_v2_1.0_224"

PRETRAINED_CKPT = ROO_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROO_PATH, 'output/trained_weights')

EVALUATE_DIR = ROO_PATH + '/output/evaluate_result_pickle/'
test_annotate_path = '/home/yjr/DataSet/VOC/VOC_test/VOC2007/Annotations'

# ------------------------------------------ Train config
RESTORE_FROM_RPN = False
IS_FILTER_OUTSIDE_BOXES = True
FIXED_BLOCKS = 1  # allow 0~3

RPN_LOCATION_LOSS_WEIGHT = 2.
RPN_CLASSIFICATION_LOSS_WEIGHT = 1.0

FAST_RCNN_LOCATION_LOSS_WEIGHT = 2.0
FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT = 1.0
RPN_SIGMA = 1.0  # 3.0
FASTRCNN_SIGMA = 1.0


MUTILPY_BIAS_GRADIENT = None   # 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = 10.0   # 10.0  if None, will not clip

EPSILON = 1e-5
MOMENTUM = 0.9
LR = 0.001  # 0.001  # 0.0003
DECAY_STEP = [50000, 70000]  # 50000, 70000
MAX_ITERATION = 100000

# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'pascal'  #'pascal'  # 'ship', 'spacenet', 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 600
IMG_MAX_LENGTH = 1024 # 1000
CLASS_NUM = 20  # 20

# --------------------------------------------- Network_config
BATCH_SIZE = 1
# INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01)
INITIALIZER = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
# BBOX_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.001)
BBOX_INITIALIZER = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
# WEIGHT_DECAY = 0.00004 if NET_NAME.startswith('Mobilenet') else 0.0001
WEIGHT_DECAY = 0.0

# ---------------------------------------------Anchor config
BASE_ANCHOR_SIZE_LIST = [256]  # can be modified
ANCHOR_STRIDE = [16]  # can not be modified in most situations
ANCHOR_SCALES = [0.5, 1., 2.0]  # [4, 8, 16, 32]
ANCHOR_RATIOS = [0.5, 1., 2.0]
ROI_SCALE_FACTORS = [10., 10., 5.0, 5.0]
ANCHOR_SCALE_FACTORS = [10., 10., 5., 5.]  # [10.0, 10.0, 5.0, 5.0]


# --------------------------------------------RPN config
KERNEL_SIZE = 3
RPN_IOU_POSITIVE_THRESHOLD = 0.7
RPN_IOU_NEGATIVE_THRESHOLD = 0.3
TRAIN_RPN_CLOOBER_POSITIVES = False

RPN_MINIBATCH_SIZE = 256
RPN_POSITIVE_RATE = 0.5
RPN_NMS_IOU_THRESHOLD = 0.7
RPN_TOP_K_NMS_TRAIN = 12000
RPN_MAXIMUM_PROPOSAL_TARIN = 2000            ##########OHEM sample in  num

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
FAST_RCNN_MINIBATCH_SIZE = 256  # 256  # if is -1, that is train with OHEM
FAST_RCNN_POSITIVE_RATE = 0.25

ADD_GTBOXES_TO_TRAIN = False



