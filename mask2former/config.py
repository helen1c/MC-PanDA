# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper

    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_panoptic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = None

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # mit backbone config
    cfg.MODEL.MIT = CN()
    cfg.MODEL.MIT.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.MIT.PATCH_SIZE = 4
    cfg.MODEL.MIT.EMBED_DIMS = [64, 128, 320, 512]
    cfg.MODEL.MIT.DEPTHS = [3, 6, 40, 3]
    cfg.MODEL.MIT.NUM_HEADS = [1, 2, 5, 8]
    cfg.MODEL.MIT.MLP_RATIOS = [4, 4, 4, 4]
    cfg.MODEL.MIT.SR_RATIOS = [8, 4, 2, 1]
    cfg.MODEL.MIT.QKV_BIAS = True
    cfg.MODEL.MIT.QK_SCALE = None
    cfg.MODEL.MIT.DROP_RATE = 0.0
    cfg.MODEL.MIT.ATTN_DROP_RATE = 0.0
    cfg.MODEL.MIT.DROP_PATH_RATE = 0.1
    cfg.MODEL.MIT.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = (
        "MultiScaleMaskedTransformerDecoder"
    )

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = [
        "res3",
        "res4",
        "res5",
    ]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75


def add_semi_sup_config(cfg):
    cfg.INPUT.CROP.RCS = False
    cfg.INPUT.CROP.RARE_CLASSES = []

    cfg.DATASETS.TRAIN_UNLABELED = ""
    cfg.UNLABELED_INPUT = CN()
    cfg.UNLABELED_INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.UNLABELED_INPUT.MAX_SIZE_TRAIN = 1333
    cfg.UNLABELED_INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.UNLABELED_INPUT.CROP = CN()
    cfg.UNLABELED_INPUT.CROP.ENABLED = True
    cfg.UNLABELED_INPUT.CROP.TYPE = "absolute"
    cfg.UNLABELED_INPUT.CROP.SIZE = (512, 1024)
    cfg.UNLABELED_INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.UNLABELED_INPUT.RANDOM_FLIP = True
    cfg.UNLABELED_INPUT.SEGMIX_ENABLED = False
    cfg.UNLABELED_INPUT.SEGMIX_PROB = 1.0
    cfg.UNLABELED_INPUT.SEPARATE_STUFF_CLASSES_MIX = False
    cfg.UNLABELED_INPUT.SEGMIX_THING_CONT_NUM = None
    cfg.UNLABELED_INPUT.STUDENT_JITTER = CN()
    cfg.UNLABELED_INPUT.STUDENT_JITTER.COLOR_JITTER = CN()
    cfg.UNLABELED_INPUT.STUDENT_JITTER.COLOR_JITTER.ENABLED = True
    cfg.UNLABELED_INPUT.STUDENT_JITTER.COLOR_JITTER.BRIGHTNESS = (0.2, 1.8)
    cfg.UNLABELED_INPUT.STUDENT_JITTER.COLOR_JITTER.CONTRAST = (0.2, 1.8)
    cfg.UNLABELED_INPUT.STUDENT_JITTER.COLOR_JITTER.SATURATION = (0.2, 1.8)
    cfg.UNLABELED_INPUT.STUDENT_JITTER.COLOR_JITTER.HUE = (-0.2, 0.2)
    cfg.UNLABELED_INPUT.STUDENT_JITTER.RANDOM_GRAYSCALE_PROB = 0.2
    cfg.UNLABELED_INPUT.STUDENT_JITTER.GAUSSIAN_BLUR_SIGMA = (0.1, 2.0)
    cfg.UNLABELED_INPUT.STUDENT_JITTER.GAUSSIAN_BLUR_PROB = 0.5

    cfg.MODEL.TEACHER = CN()
    # Different teacher configurations
    # Fixed -> set burn_in_iters to value larger than training iters, and the path to the weights
    # EMA -> set burn_in_iters to the desired value, the path to the weights (if burn_in_iters > 0) and the decay
    # student_copy (teacher == student) -> set burn_in_iters to the desired value, the path to the weights (if burn_in_iters > 0) and the ema_decay to 0

    cfg.MODEL.TEACHER.PANOPTIC_ON = True
    cfg.MODEL.TEACHER.SEMANTIC_ON = False

    cfg.MODEL.TEACHER.WEIGHTS = ""
    cfg.MODEL.TEACHER.EMA_DECAY = 0.999
    cfg.MODEL.TEACHER.FILTER_SMALL_OBJECTS_AREA_THRESHOLD = 0.0
    cfg.MODEL.TEACHER.BURN_IN_ITERS = 0
    cfg.MODEL.STUDENT = CN()
    cfg.MODEL.STUDENT.LOSSES_LABELED = ["labels", "masks"]
    cfg.MODEL.STUDENT.LOSSES_PSEUDOLABELED = ["labels", "masks"]
    cfg.MODEL.STUDENT.UNSUPERVISED_LOSS = CN()
    cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.DEEP_SUPERVISION = True
    # how many layers to exclude from deep supervision, starting from the last one (0th layer from transformer decoder)
    cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.EXCLUDE_IN_DEEP_SUPERVISION = None
    cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.CLASS_WEIGHT = 2.0
    cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.DICE_WEIGHT = 5.0
    cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.MASK_WEIGHT = 5.0
    cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.TRAIN_NUM_POINTS = 112 * 112
    cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.IMPORTANCE_SAMPLE_RATIO = 0.75
    cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.POINT_SAMPLING = CN()
    cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.POINT_SAMPLING.STRATEGY = (
        "m2f"  # "m2f_with_teacher_panoptic_uncertainty_filtering"
    )
    cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.POINT_SAMPLING.TEACHER_UNCERTAINTY_THRESHOLD = (
        None
    )
    cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.POINT_SAMPLING.TEACHER_UNCERTAINTY_TYPE = (
        "per_mask"
    )
    # this is needed when we calculate mask confidences for "per_mask" and "per_image" weight loss types
    # when 0.0, confidence for each mask will be 1.0, and 0.0 is default value
    cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.CONFIDENCE_CALCULATOR_TYPE = "per_mask"
    cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.TAU_CONFIDENCE_THRESHOLD_THINGS = 0.0
    cfg.MODEL.STUDENT.UNSUPERVISED_LOSS.TAU_CONFIDENCE_THRESHOLD_STUFF = 0.0
