# Native Python Runtime Installation

Python virtual environment is high recommended. For quick setup, we recommend to use [mambaforge](https://github.com/conda-forge/miniforge#miniforge3) as a replacement of anaconda. Replace the `mamba` command if you choose to stay with `conda`.

Create virtual environment:
```bash
mamba create -n cyto python=3.10 -y
mamba activate cyto
```

## CUDA Acceleration
In some process we can accelerate the analysis process with CUDA GPU. To achieve CUDA version compatibility **Tensorflow 2.12.0** and **pyTorch 2.0.1** must be used together with **CUDA 11.8**. 

### Dask CUDA
[Dask CUDA](https://github.com/rapidsai/dask-cuda) provides various utilities to improve deployment and management of Dask workers on CUDA-enabled systems.

We have to install through mamba/conda instead of pip for better version control.
```bash
mamba install -c rapidsai -c conda-forge -c nvidia dask-cuda cudatoolkit=11.8
```

### Tensorflow
For Stardist segmentation we recommend to use TF2 verison. Follow the instruction for [TF 2.12 installation](https://www.tensorflow.org/install/pip).

### pyTorch
For Cellpose segmentation we recommend to use pyTorch 2.0.. Follow the instruction for [pyTorch installation](https://pytorch.org/get-started/locally/).

## Required Packages

```bash
git clone git@github.com:bpi-oxford/Cytotoxicity-Pipeline.git
cd Cytotoxicity-Pipeline
pip install -r requirements.txt
```

## Fiji 
Download Fiji and install TrackMateCSVImporter from CLI:
```bash
wget https://downloads.micron.ox.ac.uk/fiji_update/mirrors/fiji-latest/fiji-linux64.zip
unzip fiji-linux64.zip
rm fiji-linux64.zip
cd /Fiji.app
./ImageJ-linux64 --headless --ij2 --update add-update-site TrackMateCSVImporter https://sites.imagej.net/TrackMateCSVImporter/
```

To configure the path of Fiji specify the path to `Fiji.app` in the line `fiji_dir` in [../examples/pipeline.yaml](../examples/pipeline.yaml)

### PyImageJ
For the integration of TrackMate, we need to embed PyImageJ for the python wrapping to Fiji.

```bash
mamba install pyimagej openjdk=8
```


