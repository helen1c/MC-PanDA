# This file is a modifies version of the original criterion.py file from the MaskFormer repository.
# The modifications are made to the SetCriterion class to include the loss for the pseudo-labels.
# Modified by Ivan Martinovic
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from torch import nn
from detectron2.layers import cat
from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from ..utils.misc import (
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)

from .criterion_utils import (
    sigmoid_ce_loss_jit,
    dice_loss_jit,
    sigmoid_ce_loss_with_pseudolabel_confidences_jit,
    dice_loss_with_pseudolabel_confidences_jit,
    calculate_uncertainty,
    get_points_with_teacher_uncertainty_filtering,
    confidence_calculator_map,
)


class SetCriterionPseudolabels(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        tau_confidence_threshold_things,
        tau_confidence_threshold_stuff,
        confidence_calculator_type,
        point_sampling_strategy,
        point_sampling_teacher_uncertainty_threshold,
        point_sampling_teacher_uncertainty_type,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        if "masks_with_conf_ppe_prob" in self.losses:
            assert self.both_ppe_probability is not None

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.tau_confidence_threshold_things = tau_confidence_threshold_things
        self.tau_confidence_threshold_stuff = tau_confidence_threshold_stuff
        self.confidence_calculator = confidence_calculator_map[
            confidence_calculator_type
        ](self.tau_confidence_threshold_things, self.tau_confidence_threshold_stuff)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.point_sampling_strategy = point_sampling_strategy
        self.point_sampling_teacher_uncertainty_threshold = (
            point_sampling_teacher_uncertainty_threshold
        )
        self.point_sampling_teacher_uncertainty_type = (
            point_sampling_teacher_uncertainty_type
        )

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"pseudo_loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]

        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "pseudo_loss_mask": sigmoid_ce_loss_jit(
                point_logits, point_labels, num_masks
            ),
            "pseudo_loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_masks_with_conf(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]

        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        loss_scaling_confidences = [t["loss_scaling_confidences"] for t in targets]
        loss_scaling_confidences = torch.cat(
            [
                confidences[tgt_idx[1][tgt_idx[0] == i]]
                for i, confidences in enumerate(loss_scaling_confidences)
            ]
        )

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            # sample point_coords
            if (
                self.point_sampling_strategy
                == "m2f_with_teacher_panoptic_uncertainty_filtering"
            ):
                if self.point_sampling_teacher_uncertainty_type == "per_image":
                    pixel_uncertainties = []
                    for t in targets:
                        if t["segment_confidence_maps"].shape[0] == 0:
                            pixel_uncertainties.append(
                                torch.zeros(
                                    (t["masks"].shape[-2], t["masks"].shape[-1]),
                                    dtype=torch.float32,
                                ).to(t["masks"])
                            )
                        else:
                            pixel_uncertainties.append(
                                -t["segment_confidence_maps"].max(0).values
                            )
                    pixel_uncertainties = torch.stack(pixel_uncertainties)
                    pixel_uncertainties_per_mask = pixel_uncertainties[tgt_idx[0]]

                elif self.point_sampling_teacher_uncertainty_type == "per_mask":
                    segment_per_pixel_logits = [
                        t["segment_per_pixel_logits"] for t in targets
                    ]
                    segment_per_pixel_logits, valid = nested_tensor_from_tensor_list(
                        segment_per_pixel_logits
                    ).decompose()
                    segment_per_pixel_logits = segment_per_pixel_logits.to(src_masks)
                    segment_per_pixel_logits = segment_per_pixel_logits[tgt_idx]
                    pixel_uncertainties_per_mask = (
                        -segment_per_pixel_logits.abs().sigmoid()
                    )

                point_coords = get_points_with_teacher_uncertainty_filtering(
                    src_masks,
                    lambda logits: calculate_uncertainty(logits),
                    pixel_uncertainties_per_mask,
                    self.point_sampling_teacher_uncertainty_threshold,
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )

            elif self.point_sampling_strategy == "m2f":
                point_coords = get_uncertain_point_coords_with_randomness(
                    src_masks,
                    lambda logits: calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "pseudo_loss_mask": sigmoid_ce_loss_with_pseudolabel_confidences_jit(
                point_logits, point_labels, loss_scaling_confidences, num_masks
            ),
            "pseudo_loss_dice": dice_loss_with_pseudolabel_confidences_jit(
                point_logits, point_labels, loss_scaling_confidences, num_masks
            ),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
            "masks_with_conf": self.loss_masks_with_conf,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def compute_loss_normalizing_factor(self, targets):
        if "masks_with_conf" in self.losses or "masks_conf_ppe_prob" in self.losses:
            self.confidence_calculator(targets)
            num_masks = sum(t["loss_scaling_confidences"].sum().item() for t in targets)
        else:
            num_masks = sum(len(t["labels"]) for t in targets)
        return num_masks

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = self.compute_loss_normalizing_factor(targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_masks
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
