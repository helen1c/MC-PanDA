# MC-PanDA Model Zoo 

## Introduction

In this file, we provide checkpoints for supervised-pretrained models on **Synthia** and **Cityscapes** (teacher initialization for the burn-in stage), and one checkpoint for each of the following domain adaptation setups:
- Synthia → Cityscapes
- Synthia → Mapillary Vistas
- Cityscapes → Foggy Cityscapes
- Cityscapes → Mapillary Vistas

For each checkpoint, we also provide a `.yaml` file in setup column, which contains all hyperparameters needed to reproduce our experiments.

#### Swin-B ImageNet Pretrained Backbone

Our paper also uses ImageNet pretrained models that are not part of Detectron2. Please refer to the [tools](https://github.com/facebookresearch/MaskFormer/tree/master/tools) to get those pretrained models.

---

## MC-PanDA Checkpoints

### Panoptic Segmentation
The paper reports results averaged over three seeds. Here, we provide a checkpoint for each source-target domain pair using a single seed. The results shown in the table correspond to these specific checkpoints and may differ from the averaged results presented in the paper, which are based on three runs. You can download `pretrained` folder from [OneDrive](https://ferhr-my.sharepoint.com/:f:/g/personal/imartinovic_fer_hr/EmPAHCBM9lJGrfkYxGpTmSwBTFA_vm9mS49uCGMRCdpobQ?e=CBz51H). 

For the **Cityscapes → Foggy** and **Cityscapes → Vistas** setups, the PQ (Panoptic Quality) is averaged over all 19 classes. In contrast, for the **SYNTHIA → (Cityscapes, Vistas)** setups, the PQ is averaged over the 16 classes available in the SYNTHIA dataset. Thus, when evaluating **Synthia → (Cityscapes, Vistas)** models, final result should be multiplied with 19/16.
<table>
<tbody>
<tr>
<th valign="bottom">Setup (cfg file)</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Iterations</th>
<th valign="bottom">PQ</th>
<th valign="bottom">SQ</th>
<th valign="bottom">RQ</th>
<th valign="bottom">Checkpoint</th>
</tr>
<!-- Supervised Rows -->
<tr>
<td align="left"><a href="configs/synthia_supervised/panoptic-segmentation/swin/maskformer2_swin-base_22k_bs4_90k_crop512x1024_sup.yaml">Synthia (supervised)</a></td>
<td align="center">Swin-B</td>
<td align="center">20k</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center"><a href="https://ferhr-my.sharepoint.com/:f:/g/personal/imartinovic_fer_hr/EuhQA1vJes5HtkcMe4uxXPUBObNJTwYT4DfcdDSEVPSigw?e=LgIfnp">model</a></td>
</tr>
<tr>
<td align="left"><a href="configs/cityscapes_supervised/panoptic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_20k_sup.yaml">Cityscapes (supervised)</a></td>
<td align="center">Swin-B</td>
<td align="center">40k</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center"><a href="https://ferhr-my.sharepoint.com/:f:/g/personal/imartinovic_fer_hr/EuPN_w-Eia9HuRlZFuICiGABm8cm32-92p_usMSwjgnzMg?e=vvjrVK">model</a></td>
</tr>
<!-- Horizontal Divider -->
<tr>
</tr>
<!-- Domain Adaptation Rows -->
<tr>
<td align="left"><a href="configs/synthia_to_city/panoptic-segmentation/swin/maskformer2_swin-base_22k_bs4_90k_crop512x1024_conf_unl.yaml">Synthia → Cityscapes</a></td>
<td align="center">Swin-B</td>
<td align="center">90k</td>
<td align="center">48.5</td>
<td align="center">77.3</td>
<td align="center">60.6</td>
<td align="center"><a href="https://ferhr-my.sharepoint.com/:f:/g/personal/imartinovic_fer_hr/EglJKft2s9pPhn6ifgaQVesBaB2GLIT9lLtKJgQj_I0Bpw?e=y64jUY">model</a></td>
</tr>
<tr>
<td align="left"><a href="configs/synthia_to_vistas/panoptic-segmentation/swin/maskformer2_unlabeled_vistas_swin-base_22k_bs4_90k_crop512x1024_conf.yaml">Synthia → Vistas</a></td>
<td align="center">Swin-B</td>
<td align="center">90k</td>
<td align="center">38.8</td>
<td align="center">71.6</td>
<td align="center">50.2</td>
<td align="center"><a href="https://ferhr-my.sharepoint.com/:f:/g/personal/imartinovic_fer_hr/Eu-yDZG9dOBFphaNWvQZAVIB2eT81W51Un_ZnFvFP9MbLw?e=QtLkMf">model</a></td>
</tr>
<tr>
<td align="left"><a href="configs/city_to_foggy/panoptic-segmentation/swin/maskformer2_swin-base_22k_bs4_90k_crop512x1024_conf.yaml">Cityscapes → Foggy</a></td>
<td align="center">Swin-B</td>
<td align="center">90k</td>
<td align="center">62.5</td>
<td align="center">82.7</td>
<td align="center">74.9</td>
<td align="center"><a href="https://ferhr-my.sharepoint.com/:f:/g/personal/imartinovic_fer_hr/EicIqx-yst9MsEWFRxjlNQcBAZbCEYSzALOWY7LV2L_lXQ?e=4mhIyz">model</a></td>
</tr>
<tr>
<td align="left"><a href="configs/city_to_vistas/panoptic-segmentation/swin/maskformer2_swin-base_22k_bs4_90k_crop512x1024_conf.yaml">Cityscapes → Vistas</a></td>
<td align="center">Swin-B</td>
<td align="center">90k</td>
<td align="center">52.0</td>
<td align="center">79.7</td>
<td align="center">64.3</td>
<td align="center"><a href="https://ferhr-my.sharepoint.com/:f:/g/personal/imartinovic_fer_hr/EoWVBF6hKnpArJpUxpYASpgBUg26oE8B61EUe7jQikDJCQ?e=NdA4mU">model</a></td>
</tr>
</tbody>
</table>

---

#### License

All models available for download through this document are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).
