## Installation

### Requirements

Our experiments were conducted within a Conda environment with the following requirements:

- **Operating System**: Linux or macOS
- **Python**: Version 3.11 or higher (earlier versions might also be compatible)
- **CUDA**: Version 12.1
- **PyTorch**: Version 2.1 or higher (earlier versions might also be compatible)
- **Torchvision**: Ensure that the Torchvision version matches the PyTorch installation. Install both together from [pytorch.org](https://pytorch.org) to ensure compatibility. Note: Verify that the PyTorch version is compatible with Detectron2.
- **Detectron2**: Follow the [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- **OpenCV**: Optional, but required for demo and visualization.

To install additional dependencies, use the following command:
```sh
pip install -r requirements.txt
```

Make sure to set up and activate your Conda environment before installing these dependencies.

**Note**: While our experiments were conducted with the specified library versions, earlier versions of Python and PyTorch might also be compatible.
[Mask2Former INSTALL.md](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md) should also work fine.

### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

### Example conda environment setup
```bash
conda create --name mc-panda python=3.11 -y
conda activate mc-panda
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
git clone https://github.com/helen1c/MC-PanDA.git
cd MC-PanDA
mkdir third-party
cd third-party
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

cd ../..
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```
