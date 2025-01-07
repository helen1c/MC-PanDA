from detectron2.engine import SimpleTrainer
import time
import torch
from detectron2.structures import Instances, BitMasks

from mask2former.utils.misc import fuse_segment_per_pixel_logits

INF_VALUE = 1000.0


class SemiSupTrainRunner(SimpleTrainer):
    def __init__(
        self,
        teacher,
        student,
        data_loader,
        optimizer,
        gather_metric_period=1,
        zero_grad_before_forward=False,
        async_write_metrics=False,
        separate_stuff_classes=False,
    ):
        super().__init__(
            student,
            data_loader,
            optimizer,
            gather_metric_period=gather_metric_period,
            zero_grad_before_forward=zero_grad_before_forward,
            async_write_metrics=async_write_metrics,
        )
        self.separate_stuff_classes = separate_stuff_classes
        self.teacher = teacher

    def obtain_segmixed_image(self, batch, classmix_box_mask):
        student_image_classmixed = batch["student_image"].clone()
        student_image_classmixed[:, classmix_box_mask] = batch["image"][
            :, classmix_box_mask
        ]

        return student_image_classmixed

    def obtain_segmixed_panseg(self, batch, classmix_box_mask):
        teacher_panseg = batch["teacher_pan_seg"].clone()
        pan_seg_gt = batch["pan_seg_gt"].to(teacher_panseg)
        max_id = teacher_panseg.max()

        pan_seg_gt[pan_seg_gt > 0] += max_id
        teacher_panseg[classmix_box_mask] = pan_seg_gt[classmix_box_mask]

        classes = []
        masks = []
        segment_confidence_maps = []
        segments_per_pixel_logits = []

        unique_ids = torch.unique(teacher_panseg)
        stuff_classes_map = {}
        for ind, segment_info in enumerate(batch["teacher_segments_info"]):
            assert ind == segment_info["id"] - 1
            if segment_info["id"] not in unique_ids:
                continue
            cat_id = segment_info["category_id"]
            if segment_info["isthing"]:
                classes.append(cat_id)
                masks.append(teacher_panseg == segment_info["id"])
                segment_confidence_maps.append(
                    batch["teacher_instances_panseg"].segment_confidence_maps[
                        segment_info["id"] - 1
                    ]
                )
                segments_per_pixel_logits.append(
                    batch["teacher_instances_panseg"].segment_per_pixel_logits[
                        segment_info["id"] - 1
                    ]
                )
            else:
                if cat_id not in stuff_classes_map:
                    stuff_classes_map[cat_id] = len(classes)
                    classes.append(cat_id)
                    masks.append(teacher_panseg == segment_info["id"])
                    segment_confidence_maps.append(
                        batch["teacher_instances_panseg"].segment_confidence_maps[
                            segment_info["id"] - 1
                        ]
                    )
                    segments_per_pixel_logits.append(
                        batch["teacher_instances_panseg"].segment_per_pixel_logits[
                            segment_info["id"] - 1
                        ]
                    )
                else:
                    masks[stuff_classes_map[cat_id]] |= (
                        teacher_panseg == segment_info["id"]
                    )
                    segment_confidence_maps[stuff_classes_map[cat_id]] = torch.max(
                        segment_confidence_maps[stuff_classes_map[cat_id]],
                        batch["teacher_instances_panseg"].segment_confidence_maps[
                            segment_info["id"] - 1
                        ],
                    )
                    segments_per_pixel_logits[stuff_classes_map[cat_id]] = (
                        fuse_segment_per_pixel_logits(
                            segments_per_pixel_logits[stuff_classes_map[cat_id]],
                            batch["teacher_instances_panseg"].segment_per_pixel_logits[
                                segment_info["id"] - 1
                            ],
                        )
                    )

        for ind, segment_info in enumerate(batch["segments_info"]):
            if segment_info["id"] + max_id not in unique_ids:
                continue
            cat_id = segment_info["category_id"]
            new_segment_id = segment_info["id"] + max_id
            is_thing = segment_info.get("isthing", None)

            mask = teacher_panseg == new_segment_id
            assert isinstance(mask, torch.Tensor)

            per_pixel_logits = torch.full_like(
                mask, fill_value=-INF_VALUE, dtype=torch.float32
            )
            per_pixel_logits[mask] = INF_VALUE

            confidence_map = mask * 1.0

            if is_thing is None:
                is_thing = cat_id >= 11
            if is_thing:
                classes.append(cat_id)
                masks.append(mask)
                segment_confidence_maps.append(confidence_map)
                segments_per_pixel_logits.append(per_pixel_logits)
            else:
                if cat_id not in stuff_classes_map or self.separate_stuff_classes:
                    if not self.separate_stuff_classes:
                        stuff_classes_map[cat_id] = len(classes)
                    classes.append(cat_id)
                    masks.append(mask)
                    segment_confidence_maps.append(confidence_map)
                    segments_per_pixel_logits.append(per_pixel_logits)
                else:
                    masks[stuff_classes_map[cat_id]] |= mask
                    segment_confidence_maps[stuff_classes_map[cat_id]] = torch.max(
                        segment_confidence_maps[stuff_classes_map[cat_id]],
                        confidence_map,
                    )
                    segments_per_pixel_logits[stuff_classes_map[cat_id]] = (
                        fuse_segment_per_pixel_logits(
                            segments_per_pixel_logits[stuff_classes_map[cat_id]],
                            mask * INF_VALUE,
                        )
                    )

        new_instances = Instances((teacher_panseg.shape[0], teacher_panseg.shape[1]))
        new_instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        if len(masks) == 0:
            new_instances.gt_masks = torch.zeros(
                (0, teacher_panseg.shape[0], teacher_panseg.shape[1])
            )
            new_instances.segment_confidence_maps = torch.zeros(
                (0, teacher_panseg.shape[0], teacher_panseg.shape[1])
            )
            new_instances.segment_per_pixel_logits = torch.zeros(
                (0, teacher_panseg.shape[0], teacher_panseg.shape[1])
            )
        else:
            masks = BitMasks(torch.stack(masks))
            new_instances.gt_masks = masks.tensor
            segment_confidence_maps = torch.stack(segment_confidence_maps)
            new_instances.segment_confidence_maps = segment_confidence_maps
            segments_per_pixel_logits = torch.stack(segments_per_pixel_logits)
            new_instances.segment_per_pixel_logits = segments_per_pixel_logits
        return new_instances

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()

        """
        If you want to do something with the losses, you can wrap the model.
        """
        with torch.no_grad():
            self.teacher.eval()
            teacher_outputs = self.teacher(data)
        for _, (batch, teacher_output) in enumerate(zip(data, teacher_outputs)):
            if self.teacher.panoptic_on:
                batch["teacher_instances_panseg"] = teacher_output["panoptic_seg"][
                    "panoptic_instances"
                ]
                if "segmix_addons" in batch:
                    clsmix_mask = batch["segmix_addons"]["segmix_mask"] == 1
                    if clsmix_mask.sum() > 0:
                        batch["student_image"] = self.obtain_segmixed_image(
                            batch=batch,
                            classmix_box_mask=clsmix_mask,
                        )
                        batch["teacher_pan_seg"] = teacher_output["panoptic_seg"][
                            "panoptic_seg"
                        ]
                        batch["teacher_segments_info"] = teacher_output["panoptic_seg"][
                            "segments_info"
                        ]
                        batch["teacher_instances_panseg"] = self.obtain_segmixed_panseg(
                            batch=batch,
                            classmix_box_mask=clsmix_mask,
                        )

        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        if not self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()
        losses.backward()
        self.after_backward()

        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

        self.optimizer.step()