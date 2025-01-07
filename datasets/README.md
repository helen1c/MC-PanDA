# Prepare Datasets for MC-PanDA 

The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  cityscapes/
  foggy_cityscapes/
  mapillary_vistas/
  synthia/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.


## Expected dataset structure for [cityscapes](https://www.cityscapes-dataset.com/downloads/): 
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
    # below are generated Cityscapes panoptic annotation
    cityscapes_panoptic_train.json
    cityscapes_panoptic_train/
    cityscapes_panoptic_val.json
    cityscapes_panoptic_val/
    cityscapes_panoptic_test.json
    cityscapes_panoptic_test/
  leftImg8bit/
    train/
    val/
    test/
```
Install cityscapes scripts by:
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

Note: to create labelTrainIds.png, first prepare the above structure, then run cityscapesescript with:
```
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```
These files are not needed for instance segmentation.

Note: to generate Cityscapes panoptic dataset, run cityscapesescript with:
```
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createPanopticImgs.py
```
These files are not needed for semantic and instance segmentation.

## Expected dataset structure for [Foggy Cityscapes](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/): 
```
foggy_cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
    # below are generated Cityscapes panoptic annotation
    cityscapes_panoptic_train.json
    cityscapes_panoptic_train/
    cityscapes_panoptic_val.json
    cityscapes_panoptic_val/
    cityscapes_panoptic_test.json
    cityscapes_panoptic_test/
  leftImg8bit_foggy/
    train/
    val/
```
Annotations remain consistent with the Cityscapes dataset, but the images have been replaced. In our experiments, we use an attenuation factor of $\beta=0.02$, following prior work.
Foggy Cityscapes is registered in [register_foggy_cityscapes_panoptic.py](../mask2former/data/datasets/register_foggy_cityscapes_panoptic.py) script.


## Expected dataset structure for [Mapillary Vistas](https://www.mapillary.com/dataset/vistas):
```
mapillary_vistas/
  training/
    images/                ### 18000 training images
      *.jpg
  validation/
    images/                ### 2000 val images
      *.jpg
    v1.2/
      labels/              ### needed for semseg
  validation19cls/         ### 2000 val 19cls remapped annotations
    *.png
    panoptic_validation_19cls.json
```
You can generate Vistas labels remapped to the 19-class Cityscapes taxonomy (validation19cls) by following the scripts and instructions in the [EDAPS repository](https://github.com/susaha/edaps?tab=readme-ov-file#setup-datasets). These scripts also produce a JSON (panoptic_validation_19cls.json) file, which can alternatively be downloaded from [OneDrive](https://ferhr-my.sharepoint.com/:f:/g/personal/imartinovic_fer_hr/Ejsd2CUypxVFuOlqI4eLCMABVD0abteYnuObrK8Oi4J-5Q?e=8cQ7UJ). Mapillary Vistas is registered in [register_mapillary_vistas_panoptic_19cls.py](../mask2former/data/datasets/register_mapillary_vistas_panoptic_19cls.py) script.

## Expected dataset structure for [SYNTHIA-RAND-CITYSCAPES](http://synthia-dataset.net/downloads/):
```
synthia/
  RAND_CITYSCAPES/
    RGB/
      *.png
    panoptic-labels-crowdth-0-for-daformer/         ### from EDAPS
      synthia_panoptic_remapped_trainid_to_id.json
      synthia_panoptic/                
        *_panoptic.png
```
You can generate SYNTHIA panoptic labels by following the scripts and instructions in the [EDAPS repository](https://github.com/susaha/edaps?tab=readme-ov-file#setup-datasets). These scripts also produce a JSON (.json) file, which can alternatively be downloaded from [OneDrive](https://ferhr-my.sharepoint.com/:f:/g/personal/imartinovic_fer_hr/Ejsd2CUypxVFuOlqI4eLCMABVD0abteYnuObrK8Oi4J-5Q?e=8cQ7UJ). Please note that Detectron2 uses IDs instead of trainIds in the annotations file, so you must replace **trainId** with **Id** during the generation process. The SYNTHIA panoptic dataset is registered in the [register_synthia_panoptic.py](../mask2former/data/datasets/register_synthia_panoptic.py) script.

---
#### DISCLAIMER: A significant portion of this file has been adapted from the original [Mask2Former repository](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md).