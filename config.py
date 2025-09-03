import albumentations as A
import cv2
import torch
import math
from albumentations.pytorch import ToTensorV2
# from utils_org import seed_everything
import json
import numpy

# DATASET = '100DOH'
# DATASET = '100DOH_9k'
DATASET = '100DOH_DL'
# DATASET = 'DL'
DATASET_OLD = DATASET
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 4
BATCH_SIZE = 64
# IMAGE_SIZE = 416
# IMAGE_SIZE = 640
IMAGE_SIZE = 416
#IMAGE_SIZE = 416
NUM_CLASSES = 7
# LEARNING_RATE = 0.0003
# LEARNING_RATE = 0.0001
LEARNING_RATE = 0.0005
# LEARNING_RATE = 0.003
# LEARNING_RATE = 0.003
# MOMENTUM = 0.937
MOMENTUM = 0.8
LOAD_DARKNET53_IMAGENET_WEIGHTS = True
# DARKNET53_IMAGENET_WEIGHTS_PATH= "imagenet_darknet53_weights/model_best.pth.tar"
DARKNET53_IMAGENET_WEIGHTS_PATH= "yolov3_pascal_78.1map.pth.tar"
# DARKNET53_IMAGENET_WEIGHTS_PATH= "PASCAL_VOC_416_YOLOV3_CKP_ORG_PRUNEDV1_EP579.pth.tar"
# DARKNET53_IMAGENET_WEIGHTS_PATH= "100doh_9k_yolov3_ckp.pth.tar"
# WEIGHT_DECAY = 0.0005
#WEIGHT_DECAY = 0.0001
WEIGHT_DECAY = 0

NUM_EPOCHS = 600
TEST_EPOCHS = 2
SAVE_EPOCHS = 3
CONF_THRESHOLD = 0.5
# CONF_THRESHOLD = 0.4
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.5
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
# CHECKPOINT_FILE = "100DOH_DL_YOLOV3_CKP_VODV3_NF2_EP35.pth.tar"
# CHECKPOINT_FILE = "DL_YOLOV3_CKP_VODV3_NF2_EP25.pth.tar"
# CHECKPOINT_FILE = "100DOH_DL_416_YOLOV3_CKP_EP25.pth.tar"
CHECKPOINT_FILE = "100DOH_DL_416_YOLOV3_CKP_PRUNEDV1_EP35.pth.tar"
# CHECKPOINT_FILE = "DL_YOLOV3_CKP_EP25.pth.tar"
RESUME = True
CHECKPOINT_FILE_SAVE_NAME = DATASET+"_"+str(IMAGE_SIZE)+"_YOLOV3_CKP"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"



DO_VOD = False
# DO_VOD = False
PREV_FRAMES_DIR = "/home/drcat/pycharm_workspace/100DOH_FRCNN_MOD_FGFA_TROIA/data/VOCdevkit2007_handobj_100K/VOC2007/DL_BINARY_VOD/ThreePrevImagesPaths"
DL_PREV_FRAMES_BASE_PATH = "/home/drcat/DATASETS/dataset_dl"
PREV_FRAMES_DIR_100DOH_PATH = "/home/drcat/DATASETS/100DOH_PREV_FRAMES_DATASET/100DOH_PREV_FRAMES_DATASET"
PREV_FRAMES_DIR_100DOH_ASSO_DICT_FPATH = "/home/drcat/DATASETS/100DOH_PREV_FRAMES_DATASET/100DOH_PREV_FRAMES_DATASET/asso_dict.json"
#PREV_FRAMES_416_FLOWS_DIR_PATH = "/home/drcat/DATASETS/100DOH_DL_OPTICAL_FLOW_DATASET_416"
PREV_FRAMES_416_FLOWS_DIR_PATH = "/media/drcat/Windows/Documents and Settings/DRCAT/Desktop/100DOH_DL_OPTICAL_FLOW_DATASET_416"
PREV_FRAMES_608_FLOWS_DIR_PATH = "/home/drcat/DATASETS/100DOH_DL_OPTICAL_FLOW_DATASET_608"
NUM_OF_SUP_FRAMES = 2
NO_PREV_FRAMES_100DOH = True


#Load prev frames association dict and store it
# with open(PREV_FRAMES_DIR_100DOH_ASSO_DICT_FPATH) as user_file:
#   prev_frames_100doh_asso_dict = json.load(user_file)

ELASTIC_TRANSFORM_PROB = 0.4

# ANCHORS = [
#     [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
#     [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
#     [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
# ]  # Note these have been rescaled to be between [0, 1]

ANCHORS = [
    [(116/IMAGE_SIZE, 90/IMAGE_SIZE), (156/IMAGE_SIZE, 198/IMAGE_SIZE), (373/IMAGE_SIZE, 326/IMAGE_SIZE)],
    [(30/IMAGE_SIZE, 61/IMAGE_SIZE), (62/IMAGE_SIZE, 45/IMAGE_SIZE), (59/IMAGE_SIZE, 119/IMAGE_SIZE)],
    [(10/IMAGE_SIZE, 13/IMAGE_SIZE), (16/IMAGE_SIZE, 30/IMAGE_SIZE), (33/IMAGE_SIZE, 23/IMAGE_SIZE)],
]  # Note these have been rescaled to be between [0, 1]


# Anchors custom 100DOH 608
#23,22, 40,36, 59,56, 91,79, 123,130, 193,152, 310,101, 201,250, 332,262
#ANCHORS = [
#    [(310/IMAGE_SIZE, 101/IMAGE_SIZE), (201/IMAGE_SIZE, 250/IMAGE_SIZE), (332/IMAGE_SIZE, 262/IMAGE_SIZE)],
#    [(91/IMAGE_SIZE, 79/IMAGE_SIZE), (123/IMAGE_SIZE, 130/IMAGE_SIZE), (193/IMAGE_SIZE, 152/IMAGE_SIZE)],
#    [(23/IMAGE_SIZE, 22/IMAGE_SIZE), (40/IMAGE_SIZE, 36/IMAGE_SIZE), (59/IMAGE_SIZE, 56/IMAGE_SIZE)],
#]  # Note these have been rescaled to be between [0, 1]


# Anchors custom 100DOH 416
#14,13, 23,22, 35,29, 43,46, 72,56, 84,99, 141,87, 145,153, 227,182
#ANCHORS = [
#    [(141/IMAGE_SIZE, 87/IMAGE_SIZE), (145/IMAGE_SIZE, 153/IMAGE_SIZE), (227/IMAGE_SIZE, 182/IMAGE_SIZE)],
#    [(43/IMAGE_SIZE, 46/IMAGE_SIZE), (72/IMAGE_SIZE, 56/IMAGE_SIZE), (84/IMAGE_SIZE, 99/IMAGE_SIZE)],
#    [(14/IMAGE_SIZE, 13/IMAGE_SIZE), (23/IMAGE_SIZE, 22/IMAGE_SIZE), (35/IMAGE_SIZE, 29/IMAGE_SIZE)],
#]  # Note these have been rescaled to be between [0, 1]

def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

scale = 1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        # A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        #A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        # A.OneOf(
        #     [
        #         A.ShiftScaleRotate(
        #             rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
        #         ),
        #         A.IAAAffine(shear=15, p=0.5, mode="constant"),
        #     ],
        #     p=1.0,
        # ),
        # A.HorizontalFlip(p=0.5),
        # A.HueSaturationValue(val_shift_limit=0.4, sat_shift_limit=0.7, hue_shift_limit=0.015, p=1.0),
        # A.Blur(p=0.1),
        # A.MotionBlur(p=0.1),
        # A.CLAHE(p=0.1),
        # A.ISONoise(p=0.3),
        # # A.Posterize(p=0.1),
        # A.ToGray(p=0.1),
        # # A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)

train_transform_100doh_vod = A.Compose (
    [

        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        # A.ElasticTransform(always_apply=True, border_mode=0, alpha=1.0, sigma=15.0, alpha_affine=15.0),
        # A.PiecewiseAffine(always_apply=True, nb_cols=3, nb_rows=3),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=1.0, label_fields=[],),
)

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

detect_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
)

LABELS_100DOH = [
    'targetobject',
    'left_hand',
    'right_hand'
]

LABELS_100DOH_COLORS = [
    (0, 255, 255),
    (0, 0, 255),
    (0, 255, 0)
]

