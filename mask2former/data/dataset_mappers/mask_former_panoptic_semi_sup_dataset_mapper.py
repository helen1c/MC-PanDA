# Copyright (c) Facebook, Inc. and its affiliates.
import copy
from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import detection_utils as utils, DatasetCatalog
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances
from detectron2.projects.point_rend import ColorAugSSDTransform

from .mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper
import torchvision.transforms.v2 as tv_tv2
from random import choice
import PIL
import random
from PIL import ImageFilter
from ..class_uniform_crop import ClassUniformCrop
from panopticapi.utils import rgb2id

class PanSegAugInput(T.AugInput):
    def __init__(
        self,
        image: np.ndarray,
        *,
        boxes: Optional[np.ndarray] = None,
        sem_seg: Optional[np.ndarray] = None,
        pan_seg: Optional[np.ndarray] = None,
        segments_info: Optional[list] = None,
    ):
        """
        Args:
            image (ndarray): (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
                floating point in range [0, 1] or [0, 255]. The meaning of C is up
                to users.
            boxes (ndarray or None): Nx4 float32 boxes in XYXY_ABS mode
            sem_seg (ndarray or None): HxW uint8 semantic segmentation mask. Each element
                is an integer label of pixel.
        """
        super().__init__(image, boxes=boxes, sem_seg=sem_seg)
        self.pan_seg = pan_seg
        self.segments_info = segments_info

    def transform(self, tfm) -> None:
        """
        In-place transform all attributes of this class.

        By "in-place", it means after calling this method, accessing an attribute such
        as ``self.image`` will return transformed data.
        """
        self.image = tfm.apply_image(self.image)
        if self.boxes is not None:
            self.boxes = tfm.apply_box(self.boxes)
        if self.sem_seg is not None:
            self.sem_seg = tfm.apply_segmentation(self.sem_seg)
        if self.pan_seg is not None:
            self.pan_seg = tfm.apply_segmentation(self.pan_seg)

__all__ = ["MaskFormerPanopticSemiSupDatasetMapper"]

def obtain_segmix_mask(
    classes, masks, p=1.0, mask_shape=(512, 1024), thing_cont_num=None
):
    final_mask = torch.zeros(mask_shape)
    if random.random() > p or len(masks) == 0:
        return final_mask

    masks = np.stack(masks)
    n_segments = classes.shape[0]
    indices_choice = np.random.choice(
        n_segments, int((n_segments + n_segments % 2) / 2), replace=False
    )  # random permutation
    if thing_cont_num is not None:
        indices_choice = indices_choice[classes[indices_choice] >= thing_cont_num]
    masks_sum = np.sum(masks[indices_choice], axis=0)
    return final_mask if len(masks_sum) == 0 else masks_sum


class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class MaskFormerPanopticSemiSupDatasetMapper(MaskFormerSemanticDatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        category_rep=None,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        unlabeled_dataset,
        unlabeled_transform,
        student_jitter,
        segmix_enabled,
        segmix_prob,
        rcs_crop,
        thing_cont_num=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        super().__init__(
            is_train,
            augmentations=augmentations,
            image_format=image_format,
            ignore_label=ignore_label,
            size_divisibility=size_divisibility,
        )
        self.category_rep = category_rep
        self.unlabeled_dataset = unlabeled_dataset
        self.unlabeled_transform = unlabeled_transform
        self.student_jitter = student_jitter
        self.segmix_enabled = segmix_enabled
        self.segmix_prob = segmix_prob
        self.rcs_crop = rcs_crop
        self.thing_cont_num = thing_cont_num

        assert self.segmix_prob >= 0.0 and self.segmix_prob <= 1.0

        

    @classmethod
    def from_config(cls, cfg, is_train=True, category_rep=None):
        ret = super().from_config(cfg, is_train)
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            if cfg.INPUT.CROP.RCS:
                augs.append(
                    ClassUniformCrop(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                        cfg.INPUT.CROP.RARE_CLASSES,
                        category_rep,
                    )
                )
                ret["rcs_crop"] = True
            else:
                augs.append(
                    T.RandomCrop_CategoryAreaConstraint(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                        cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                        cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    )
                )
                ret["rcs_crop"] = False

        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())
        ret["augmentations"] = augs
        
        unlabeled_dataset = DatasetCatalog.get(cfg.DATASETS.TRAIN_UNLABELED)
        # different supervised images, one for segmix another for learning
        # assert len(cfg.DATASETS.TRAIN) == 1
        # labeled_dataset = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        ret["unlabeled_dataset"] = unlabeled_dataset
        # ret["labeled_dataset"] = labeled_dataset
        augs = [
            T.ResizeShortestEdge(
                cfg.UNLABELED_INPUT.MIN_SIZE_TRAIN,
                cfg.UNLABELED_INPUT.MAX_SIZE_TRAIN,
                cfg.UNLABELED_INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.UNLABELED_INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.UNLABELED_INPUT.CROP.TYPE,
                    cfg.UNLABELED_INPUT.CROP.SIZE,
                    cfg.UNLABELED_INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        augs.append(T.RandomFlip())
        ret["unlabeled_transform"] = augs
        student_transform = []
        if cfg.UNLABELED_INPUT.STUDENT_JITTER.COLOR_JITTER.ENABLED:
            student_transform.append(
                tv_tv2.ColorJitter(
                    brightness=cfg.UNLABELED_INPUT.STUDENT_JITTER.COLOR_JITTER.BRIGHTNESS,
                    contrast=cfg.UNLABELED_INPUT.STUDENT_JITTER.COLOR_JITTER.CONTRAST,
                    saturation=cfg.UNLABELED_INPUT.STUDENT_JITTER.COLOR_JITTER.SATURATION,
                    hue=cfg.UNLABELED_INPUT.STUDENT_JITTER.COLOR_JITTER.HUE,
                )
            )
            student_transform.append(
                tv_tv2.RandomGrayscale(
                    p=cfg.UNLABELED_INPUT.STUDENT_JITTER.RANDOM_GRAYSCALE_PROB
                )
            )
            student_transform.append(
                tv_tv2.RandomApply(
                    [
                        GaussianBlur(
                            cfg.UNLABELED_INPUT.STUDENT_JITTER.GAUSSIAN_BLUR_SIGMA
                        )
                    ],
                    p=cfg.UNLABELED_INPUT.STUDENT_JITTER.GAUSSIAN_BLUR_PROB,
                )
            )

        ret["student_jitter"] = tv_tv2.Compose(student_transform)
        ret["segmix_enabled"] = cfg.UNLABELED_INPUT.SEGMIX_ENABLED
        ret["segmix_prob"] = cfg.UNLABELED_INPUT.SEGMIX_PROB
        ret["thing_cont_num"] = cfg.UNLABELED_INPUT.SEGMIX_THING_CONT_NUM
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert (
            self.is_train
        ), "MaskFormerPanopticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        unlabeled_image_path = choice(self.unlabeled_dataset)["file_name"]
        unlabeled_image = utils.read_image(unlabeled_image_path, format=self.img_format)

        # semantic segmentation
        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype(
                "double"
            )
        else:
            sem_seg_gt = None

        # panoptic segmentation
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]
        else:
            pan_seg_gt = None
            segments_info = None

        if pan_seg_gt is None:
            raise ValueError(
                "Cannot find 'pan_seg_file_name' for panoptic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        aug_input = PanSegAugInput(image, sem_seg=sem_seg_gt, pan_seg=pan_seg_gt, segments_info=segments_info)
        # aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        if sem_seg_gt is not None:
            sem_seg_gt = aug_input.sem_seg

        unsup_aug_input = T.AugInput(unlabeled_image)
        unsup_aug_input, unsup_transforms = T.apply_transform_gens(
            self.unlabeled_transform, unsup_aug_input
        )
        unlabeled_image = unsup_aug_input.image
        teacher_image = unlabeled_image
        student_image = self.transform_student_image(copy.deepcopy(unlabeled_image))
        # apply the same transformation to panoptic segmentation
        pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)
        pan_seg_gt = rgb2id(pan_seg_gt)

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        teacher_image = torch.as_tensor(
            np.ascontiguousarray(teacher_image.transpose(2, 0, 1))
        )
        student_image = torch.as_tensor(
            np.ascontiguousarray(student_image.transpose(2, 0, 1))
        )

        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))

        if self.size_divisibility is not None:
            size_divisibility_h = self.size_divisibility[0]
            size_divisibility_w = self.size_divisibility[1]
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                size_divisibility_w - image_size[1],
                0,
                size_divisibility_h - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(
                    sem_seg_gt, padding_size, value=self.ignore_label
                ).contiguous()
            pan_seg_gt = F.pad(
                pan_seg_gt, padding_size, value=0
            ).contiguous()  # 0 is the VOID panoptic label

            unlabeled_image_size = (teacher_image.shape[-2], teacher_image.shape[-1])

            padding_size = [
                0,
                size_divisibility_w - unlabeled_image_size[1],
                0,
                size_divisibility_h - unlabeled_image_size[0],
            ]
            teacher_image = F.pad(teacher_image, padding_size, value=128).contiguous()
            student_image = F.pad(student_image, padding_size, value=128).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image
        dataset_dict["teacher_image"] = teacher_image
        dataset_dict["student_image"] = student_image

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError(
                "Pemantic segmentation dataset should not have 'annotations'."
            )

        # Prepare per-category binary masks
        pan_seg_gt = pan_seg_gt.numpy()
        instances = Instances(image_shape)
        classes = []
        masks = []
        for segment_info in segments_info:
            if not segment_info["iscrowd"]:
                class_id = segment_info["category_id"]
                mask = pan_seg_gt == segment_info["id"]
                if mask.sum() > 0:
                    classes.append(class_id)
                    masks.append(mask)

        classes = np.array(classes)

        if self.segmix_enabled:
            dataset_dict["segmix_addons"] = {
                "segmix_mask": obtain_segmix_mask(
                    classes,
                    masks,
                    self.segmix_prob,
                    mask_shape=(teacher_image.shape[-2], teacher_image.shape[-1]),
                    thing_cont_num=self.thing_cont_num,
                ),
            }

        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros(
                (0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1])
            )
        else:
            masks = BitMasks(
                torch.stack(
                    [torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks]
                )
            )
            instances.gt_masks = masks.tensor

        if self.segmix_enabled:
            dataset_dict["pan_seg_gt"] = torch.as_tensor(
                np.ascontiguousarray(pan_seg_gt)
            )
        dataset_dict["instances"] = instances
        return dataset_dict

    def transform_student_image(self, student_image):
        if self.img_format == "BGR":
            student_image = student_image[:, :, ::-1]
        student_image = self.student_jitter(PIL.Image.fromarray(student_image))
        student_image = np.array(student_image)
        if self.img_format == "BGR":
            student_image = student_image[:, :, ::-1]
        return student_image