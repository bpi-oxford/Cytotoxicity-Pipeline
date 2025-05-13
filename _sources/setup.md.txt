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
For Stardist segmentation we recommend to use TF2 version. Follow the instruction for [TF 2.12 installation](https://www.tensorflow.org/install/pip).

### pyTorch
For Cellpose segmentation we recommend to use pyTorch 2.0.. Follow the instruction for [pyTorch installation](https://pytorch.org/get-started/locally/).


### CUDA and CUDNN Versions
❗The conda environment file automatically choose the compatible version between Tensorflow and pyTorch under same CUDA (11.8) and CUDNN (8.6) settings. If you find CUDA or CUDNN version inconsistency by faulty loading the machine base CUDA libraries, use the following Conda virtual environment setting to override the system-wide paths:
```bash
mamba activate cyto
mamba install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Verify CUDA and CUDNN
python3 -c "import tensorflow as tf; input_shape = (4, 28, 28, 3); x = tf.random.normal(input_shape); y = tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=input_shape[1:])(x); print(y.shape)"
```

### OpenMPI
#### Mac OS
```bash
brew install openmpi
```

## Required Packages

```bash
git clone git@github.com:bpi-oxford/Cytotoxicity-Pipeline.git
cd Cytotoxicity-Pipeline
pip install -r requirements.txt
```

If error `x86_64-conda_cos6-linux-gnu-c++ command not found` appears during mpi4py installation you will need specific GCC compiler within conda environment:
```bash
mamba install gxx_linux-64
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


