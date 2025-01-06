## Getting Started with MC-PanDA

All our experiments can be reproduced using a single A100-40 GPU card. 

### Configuration files
All necessary configuration files are located in the [configs](./configs/) directory.

### Running supervised pretraining
To run the training script for SYNTHIA supervised pretraining, use the following command: 

```bash
python train_net.py --config-file configs/synthia_supervised/panoptic-segmentation/swin/maskformer2_swin-base_22k_bs4_90k_crop512x1024_sup.yaml
```

For **Cityscapes supervised pretraining**, you only need to change the configuration file to:

```yaml
configs/cityscapes_supervised/panoptic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_20k_sup.yaml
```

### Running semi-supervised training
To perform semi-supervised training on the **SYNTHIA to Cityscapes** setup, use the following command:

```bash
python train_net_semi_sup.py --config-file configs/synthia_to_city/panoptic-segmentation/swin/maskformer2_swin-base_22k_bs4_90k_crop512x1024_conf_unl.yaml
```

For other setups, the process is similar. Configuration files for each setup can be found in the [Model Zoo](./MODEL_ZOO.md).

### Evaluation

To evaluate trained models (checkpoints are available in the [MODEL ZOO](./MODEL_ZOO.md)), use the following command:

```bash
python train_net_semi_sup.py --eval-only --config-file 'path_to_config_file' MODEL.WEIGHTS 'path_to_checkpoint'
```

Replace `'path_to_config_file'` and `'path_to_checkpoint'` with the appropriate configuration file and checkpoint paths for your evaluation.