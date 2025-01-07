# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config, add_semi_sup_config

from .data.dataset_mappers.mask_former_panoptic_dataset_mapper import (
    MaskFormerPanopticDatasetMapper,
)
from .data.dataset_mappers.mask_former_panoptic_semi_sup_dataset_mapper import (
    MaskFormerPanopticSemiSupDatasetMapper,
)

from .evaluation import COCOPanopticEvaluatorD2Modified

# models
from .maskformer_model import MaskFormer
from .maskformer_student import MaskFormerStudent
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
# from .evaluation.instance_evaluation import InstanceSegEvaluator
