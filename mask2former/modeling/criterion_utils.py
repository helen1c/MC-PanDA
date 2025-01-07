# Created by Ivan Martinovic
"""
MaskFormer criterion utils.
"""

import torch
import torch.nn.functional as F
from detectron2.layers import cat
from detectron2.projects.point_rend.point_features import (
    point_sample,
)


class DefaultConfidenceCalculator:
    def __init__(
        self, tau_confidence_threshold_things, tau_confidence_threshold_stuff
    ) -> None:
        self.tau_things = tau_confidence_threshold_things
        self.tau_stuff = tau_confidence_threshold_stuff

    def __call__(self, targets):
        for t in targets:

            loss_scaling_confidences_things = (
                (t["segment_confidence_maps"] * t["masks"]) > self.tau_things
            ).sum((-2, -1)) / t["masks"].sum((-2, -1))
            loss_scaling_confidences_stuff = (
                (t["segment_confidence_maps"] * t["masks"]) > self.tau_stuff
            ).sum((-2, -1)) / t["masks"].sum((-2, -1))

            # FIXME currently hardcoded for cityscapes taxonomy
            # for sure there is better way to achieve the same
            t["loss_scaling_confidences"] = torch.where(
                t["labels"] < 11,
                loss_scaling_confidences_stuff,
                loss_scaling_confidences_things,
            )


class PerImageConfidenceCalculator:
    def __init__(
        self, tau_confidence_threshold_things, tau_confidence_threshold_stuff
    ) -> None:
        self.tau_things = tau_confidence_threshold_things
        self.tau_stuff = tau_confidence_threshold_stuff
        assert self.tau_things == self.tau_stuff
        self.tau = self.tau_things
        print("Per image confidence calculator")

    def __call__(self, targets):
        for t in targets:
            t["loss_scaling_confidences"] = (
                (t["segment_confidence_maps"] * t["masks"]) > self.tau
            ).sum((-2, -1)) / t["masks"].sum((-2, -1))
            uncertain_masks_bitmask = t["loss_scaling_confidences"] != 1.0

            if uncertain_masks_bitmask.any():
                t["loss_scaling_confidences"][uncertain_masks_bitmask] = (
                    t["segment_confidence_maps"][uncertain_masks_bitmask].max(0).values
                    > self.tau
                ).sum() / (t["masks"].shape[1] * t["masks"].shape[2])


confidence_calculator_map = {
    "per_mask": DefaultConfidenceCalculator,
    "per_image": PerImageConfidenceCalculator,
}


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def dice_loss_with_pseudolabel_confidences(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_scaling_confidences: torch.Tensor,
    num_masks: float,
):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return (loss * loss_scaling_confidences).sum() / num_masks


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


def sigmoid_ce_loss_with_pseudolabel_confidences(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_scaling_confidences: torch.Tensor,
    num_masks: float,
):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return (loss.mean(1) * loss_scaling_confidences).sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule
sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)  # type: torch.jit.ScriptModule

dice_loss_with_pseudolabel_confidences_jit = dice_loss_with_pseudolabel_confidences
sigmoid_ce_loss_with_pseudolabel_confidences_jit = (
    sigmoid_ce_loss_with_pseudolabel_confidences
)


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def get_points_with_teacher_uncertainty_filtering(
    coarse_logits,
    uncertainty_func,
    teacher_uncertainty,
    teacher_uncertainty_threshold,
    num_points,
    oversample_ratio,
    importance_sample_ratio,
):
    """
    Modified from detectron2.projects.point_rend.point_features.get_uncertain_point_coords
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)

    point_teacher_uncertainties = point_sample(
        teacher_uncertainty.unsqueeze(1), point_coords, align_corners=False
    )

    point_uncertainties = uncertainty_func(point_logits)

    point_uncertainties[point_teacher_uncertainties > teacher_uncertainty_threshold] = (
        -torch.inf
    )

    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(
        num_boxes, dtype=torch.long, device=coarse_logits.device
    )
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(
                    num_boxes, num_random_points, 2, device=coarse_logits.device
                ),
            ],
            dim=1,
        )
    return point_coords
