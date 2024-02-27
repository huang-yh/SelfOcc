<!-- template from bevformer -->
# Step-by-step installation instructions

The following configuration of conda environment is tested on RTX3090, and we also ran our code on RTX4090 with torch==2.0.0+cu118.


**a. Create a conda virtual environment and activate it.**
```shell
conda create -n selfocc python=3.8.16 -y
conda activate selfocc
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# Recommended torch>=1.12.1
```

**c. Install mmcv, mmdet, mmseg, mmdet3d.**
```shell
pip install -U openmim  # Following https://mmcv.readthedocs.io/en/latest/get_started/installation.html
mim install mmcv==2.0.1
mim install mmdet==3.0.0 # Following https://mmdetection.readthedocs.io/en/latest/get_started.html
pip install "mmsegmentation==1.0.0" # Following https://mmsegmentation.readthedocs.io/en/latest/get_started.html
mim install "mmdet3d==1.1.1" # Following https://mmdetection3d.readthedocs.io/en/latest/get_started.html
```

**d. Install sdfstudio.**

We develop the SDF rendering part of our code based on [sdfstudio](https://github.com/autonomousvision/sdfstudio). Also inside our adapted sdfstudio, we use code from [cuda_gridsample_grad2](https://github.com/AliaksandrSiarohin/cuda-gridsample-grad2) for grid sample implimentation with second order derivative support. Many thanks to them.
```shell
git clone --recursive git@github.com:huang-yh/sdfstudio.git
cd sdfstudio
pip install --upgrade pip setuptools
pip install -e .
# install tab completion
ns-install-cli
```

**e. Install OpenSeeD.**

We use the open-vocabulary 2D segmentor [OpenSeeD](https://github.com/IDEA-Research/OpenSeeD) to generate 2D semantic maps for supervision, and install its git repo as a package. Many thanks to them.
```shell
git clone git@github.com:huang-yh/OpenSeeD.git
python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
cd OpenSeeD & python -m pip install -r requirements.txt
pip install -e .
```

**f. Install other packages and deal with package versions.**
```shell
pip install pillow==8.4.0 typing_extensions==4.8.0 torchmetrics==0.9.3 timm==0.9.2
```


**g. Clone SelfOcc.**
```
git clone https://github.com/huang-yh/SelfOcc.git
```

**h. Prepare pretrained models.**
```shell
cd SelfOcc
mkdir ckpts & cd ckpts
wget https://download.pytorch.org/models/resnet50-0676ba61.pth
wget https://github.com/IDEA-Research/OpenSeeD/releases/download/openseed/model_state_dict_swint_51.2ap.pt
```
