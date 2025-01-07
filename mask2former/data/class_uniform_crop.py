"""
Created by Ivan Martinovic
"""

import random
import numpy as np
from fvcore.transforms.transform import CropTransform
from detectron2.data.transforms.augmentation import Augmentation
from panopticapi.utils import rgb2id
import logging

logger = logging.getLogger(__name__)


class ClassUniformCrop(Augmentation):
    """
    Favors crops containing rare classes. Use only with RepeatFactorTrainingSampler.
    """

    def __init__(self, crop_type: str, crop_size, rare_classes=None, category_rep=None):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
            crop_size (tuple[float, float]): two floats, explained below.
        - "relative": crop a (H * crop_size[0], W * crop_size[1]) region from an input image of
          size (H, W). crop size should be in (0, 1]
        - "relative_range": uniformly sample two values from [crop_size[0], 1]
          and [crop_size[1]], 1], and use them as in "relative" crop type.
        - "absolute" crop a (crop_size[0], crop_size[1]) region from input image.
          crop_size must be smaller than the input image size.
        - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
          [crop_size[0], min(H, crop_size[1])] and W_crop in [crop_size[0], min(W, crop_size[1])].
          Then crop a region (H_crop, W_crop).
        """
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]
        self._init(locals())
        assert category_rep is not None
        self.category_rep = category_rep
        if rare_classes is None or len(rare_classes) == 0:
            self.rare_classes = [
                class_id
                for class_id in category_rep.keys()
                if category_rep[class_id] > 1.0
            ]
        else:
            self.rare_classes = []

        logger.info(f"Rare classes indices: {self.rare_classes}")

        self.p_true_random_crop = 0.0
        self.crop_size = crop_size

    def set_category_rep(self, category_rep):
        self.category_rep = category_rep

    def get_transform(self, image, pan_seg, segments_info):
        h, w = image.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        segment = self._choose_target_segment(segments_info)
        if segment is None:
            h0, w0 = self._rand_location(h, w, croph, cropw)
        else:
            pan_seg = rgb2id(pan_seg)
            segment_binary_mask = pan_seg == segment["id"]
            h0, w0 = self._choose_target_crop(h, w, croph, cropw, segment_binary_mask)

        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(
            self
        )
        return CropTransform(w0, h0, cropw, croph)

    def _choose_target_segment(self, segments_info):
        if segments_info is None:
            return None
        assert self.category_rep is not None

        candidate_segments = []
        candidate_weights = []
        for segment_info in segments_info:
            if segment_info["category_id"] in self.rare_classes:
                candidate_segments.append(segment_info)
                candidate_weights.append(self.category_rep[segment_info["category_id"]])

        if len(candidate_segments) == 0:
            return None

        return random.choices(candidate_segments, weights=candidate_weights)[0]

    def _choose_target_crop(self, h, w, croph, cropw, segment_binary_mask):
        if segment_binary_mask is not None:
            segment_binary_mask_sum = segment_binary_mask.sum()

            if (
                segment_binary_mask_sum > 0.0
                and random.uniform(0, 1) > self.p_true_random_crop
            ):
                for _ in range(10):
                    h0, w0 = self._rand_location(h, w, croph, cropw)
                    h1, w1 = h0 + croph, w0 + cropw
                    crop_binary_mask_sum = segment_binary_mask[h0:h1, w0:w1].sum()
                    if crop_binary_mask_sum / segment_binary_mask_sum > 0.1:
                        return h0, w0

                return h0, w0
        return self._rand_location(h, w, croph, cropw)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width
        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.randint(
                min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1
            )
            cw = np.random.randint(
                min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1
            )
            return ch, cw
        else:
            raise NotImplementedError("Unknown crop type {}".format(self.crop_type))

    def _rand_location(self, h, w, croph, cropw):
        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        return h0, w0
