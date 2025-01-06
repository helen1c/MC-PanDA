# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import io
import itertools
import json
import logging
import os
import tempfile
from collections import OrderedDict
from typing import Optional
from tabulate import tabulate

from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.evaluation import COCOPanopticEvaluator

logger = logging.getLogger(__name__)


class COCOPanopticEvaluatorD2Modified(COCOPanopticEvaluator):
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(
        self,
        dataset_name: str,
        output_dir: Optional[str] = None,
        show_per_class_results=True,
    ):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        super().__init__(dataset_name, output_dir)
        self.show_per_class_results = show_per_class_results
        logger.info("Creating COCOPanopticEvaluatorD2Modified")

        assert self._metadata.stuff_classes == self._metadata.thing_classes, (
            "The current implementation of COCOPanopticEvaluatorD2Modified assumes "
            "that stuff classes and thing classes are the same. "
            "Please use COCOPanopticEvaluator if you have different classes."
        )

        self.classes_dict = {}
        self.class_names = self._metadata.stuff_classes
        self.classes_dict.update(self._metadata.thing_dataset_id_to_contiguous_id)
        self.classes_dict.update(self._metadata.stuff_dataset_id_to_contiguous_id)

    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            from panopticapi.evaluation import pq_compute

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        for cls, entry in pq_res["per_class"].items():
            res[f"PQ_{cls}_{self.class_names[self.classes_dict[cls]]}"] = (
                100 * entry["pq"]
            )
            res[f"SQ_{cls}_{self.class_names[self.classes_dict[cls]]}"] = (
                100 * entry["sq"]
            )
            res[f"RQ_{cls}_{self.class_names[self.classes_dict[cls]]}"] = (
                100 * entry["rq"]
            )

        results = OrderedDict({"panoptic_seg": res})
        _print_panoptic_results(
            pq_res, class_names=self.class_names, classes_dict=self.classes_dict
        )

        return results


def _print_panoptic_results(pq_res, class_names, classes_dict):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = (
            [name]
            + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]]
            + [pq_res[name]["n"]]
        )
        data.append(row)

    table = tabulate(
        data,
        headers=headers,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    logger.info("Panoptic Evaluation Results:\n" + table)
    headers = ["", "PQ", "SQ", "RQ"]
    per_class_data = []

    for cls, entry in pq_res["per_class"].items():
        row = [class_names[classes_dict[cls]]] + [
            100 * entry[k] for k in ["pq", "sq", "rq"]
        ]
        per_class_data.append(row)

    table = tabulate(
        per_class_data,
        headers=headers,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    logger.info("Per-class Results:\n" + table)


if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger

    logger = setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json")
    parser.add_argument("--gt-dir")
    parser.add_argument("--pred-json")
    parser.add_argument("--pred-dir")
    args = parser.parse_args()

    from panopticapi.evaluation import pq_compute

    with contextlib.redirect_stdout(io.StringIO()):
        pq_res = pq_compute(
            args.gt_json,
            args.pred_json,
            gt_folder=args.gt_dir,
            pred_folder=args.pred_dir,
        )
        _print_panoptic_results(pq_res)
