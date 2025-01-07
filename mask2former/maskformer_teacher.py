# Modified version of maskformer_model.py
# Modified by Ivan Martinovic

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from torch.nn.parallel import DistributedDataParallel

from copy import deepcopy, copy

from .utils.misc import fuse_segment_per_pixel_logits


@META_ARCH_REGISTRY.register()
class MaskFormerTeacher(nn.Module):
    """
    Adaptation of MaskFormer for semi-supervised learning. Teacher class.
    """

    def __init__(
        self,
        maskformer,
        ema_decay,
        burn_in_iters,
        panoptic_on=True,
    ):
        """
        Args:
            maskformer: a MaskFormer instance
            type: type of teacher model, ema, fixed or student_copy
        """
        super().__init__()
        self.ema_decay = ema_decay
        self.burn_in_iters = burn_in_iters
        self.init_modules(maskformer, use_deepcopy=True)
        self.convert_weights_to_leaf_nodes()

        self.num_queries = maskformer.num_queries
        self.overlap_threshold = maskformer.overlap_threshold
        self.object_mask_threshold = maskformer.object_mask_threshold
        self.metadata = maskformer.metadata
        self.size_divisibility = maskformer.size_divisibility
        self.sem_seg_postprocess_before_inference = (
            maskformer.sem_seg_postprocess_before_inference
        )
        self.register_buffer(
            "pixel_mean", torch.Tensor(maskformer.pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(maskformer.pixel_std).view(-1, 1, 1), False
        )

        self.panoptic_on = panoptic_on
        # currently not supported
        self.semantic_on = False
        self.instance_on = False
        self.test_topk_per_image = maskformer.test_topk_per_image

        assert self.sem_seg_postprocess_before_inference

    def init_modules(self, maskformer_model, use_deepcopy):
        if use_deepcopy:
            self.backbone = deepcopy(maskformer_model.backbone)
            self.sem_seg_head = deepcopy(maskformer_model.sem_seg_head)
        else:
            self.backbone = copy(maskformer_model.backbone)
            self.sem_seg_head = copy(maskformer_model.sem_seg_head)

    def convert_weights_to_leaf_nodes(self):
        for name, param in self.named_parameters():
            param.detach()
            param.requires_grad = False

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = [x["teacher_image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            raise NotImplementedError("MaskFormerTeacher is only used for inference.")
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height, width = image_size
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                        mask_cls_result, mask_pred_result, input_per_image
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def panoptic_inference(self, mask_cls, mask_pred, input_per_image):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred_logits = mask_pred
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (
            scores > self.object_mask_threshold
        )
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_pred_logits = mask_pred_logits[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []
        instances = Instances((h, w))
        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            instances.gt_masks = torch.zeros((0, h, w))
            instances.gt_classes = torch.tensor([], dtype=torch.int64)
            instances.segment_confidence_maps = torch.zeros(
                (0, h, w), dtype=torch.float32
            )
            instances.segment_per_pixel_logits = torch.zeros(
                (0, h, w), dtype=torch.float32
            )
            ret = {
                "prob_masks": cur_prob_masks,
                "panoptic_seg": panoptic_seg,
                "segments_info": segments_info,
                "panoptic_instances": instances,
            }
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = (
                    pred_class
                    in self.metadata.thing_dataset_id_to_contiguous_id.values()
                )
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)][0]
                            segments_info[stuff_memory_list[int(pred_class)][1]][
                                "segment_confidence_map"
                            ] = torch.max(
                                segments_info[stuff_memory_list[int(pred_class)][1]][
                                    "segment_confidence_map"
                                ],
                                cur_prob_masks[k],
                            )
                            segments_info[stuff_memory_list[int(pred_class)][1]][
                                "segment_per_pixel_logits"
                            ] = fuse_segment_per_pixel_logits(
                                segments_info[stuff_memory_list[int(pred_class)][1]][
                                    "segment_per_pixel_logits"
                                ],
                                cur_mask_pred_logits[k],
                            )

                        else:
                            stuff_memory_list[int(pred_class)] = (
                                current_segment_id + 1,
                                len(segments_info),
                            )

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                            "segment_confidence_map": cur_prob_masks[k],
                            "segment_per_pixel_logits": cur_mask_pred_logits[k],
                        }
                    )

            classes = []
            masks = []
            segment_confidence_maps = []
            segment_per_pixel_logits = []

            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                classes.append(class_id)
                masks.append(panoptic_seg == segment_info["id"])
                segment_confidence_maps.append(
                    segment_info.pop("segment_confidence_map")
                )
                segment_per_pixel_logits.append(
                    segment_info.pop("segment_per_pixel_logits")
                )

            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, h, w))
                instances.segment_confidence_maps = torch.zeros(
                    (0, h, w), dtype=torch.float32
                )
                instances.segment_per_pixel_logits = torch.zeros(
                    (0, h, w), dtype=torch.float32
                )
            else:
                # Maybe make a deepcopy of masks tensors?
                masks = BitMasks(torch.stack(masks))
                instances.gt_masks = masks.tensor
                instances.segment_confidence_maps = torch.stack(segment_confidence_maps)
                instances.segment_per_pixel_logits = torch.stack(
                    segment_per_pixel_logits
                )
            ret = {
                "panoptic_seg": panoptic_seg,
                "segments_info": segments_info,
                "panoptic_instances": instances,
                "prob_masks": cur_prob_masks,
            }
        return ret

    def post_optim_step(self, student, iter):
        if isinstance(student, DistributedDataParallel):
            student = student.module
        # Burn in phase is finished...
        if iter > 0 and iter == self.burn_in_iters:
            # Copy weights from student to teacher
            self.init_modules(student, use_deepcopy=True)
            self.convert_weights_to_leaf_nodes()
            print("End of the burn-in phase. Copying weights from student to teacher.")
        elif iter > self.burn_in_iters:
            ema_decay = (
                min(1 - (1 / (iter + 1)), self.ema_decay)
                if self.burn_in_iters == 0
                else self.ema_decay
            )
            # self.init_modules(student, use_deepcopy=True)
            self.ema_update_weights(student, ema_decay=ema_decay)

    def ema_update_weights(self, student, ema_decay):
        # Update parameters.
        with torch.no_grad():
            for ema_param, param in zip(self.parameters(), student.parameters()):
                if not param.data.shape:  # scalar tensor
                    ema_param.data = (
                        ema_decay * ema_param.data + (1 - ema_decay) * param.data
                    )
                else:
                    ema_param.data[:] = (
                        ema_decay * ema_param[:].data[:]
                        + (1 - ema_decay) * param[:].data[:]
                    )

            for ema_buffer, buffer in zip(self.buffers(), student.buffers()):
                if not buffer.data.shape:
                    ema_buffer.data = (
                        ema_decay * ema_buffer.data + (1 - ema_decay) * buffer.data
                    )
                else:
                    ema_buffer.data[:] = (
                        ema_decay * ema_buffer[:].data[:]
                        + (1 - ema_decay) * buffer[:].data[:]
                    )
