# Cytotoxicity-Pipeline
A pipelined workflow for confocal cytotoxicity data.

Users may provide a YAML sript for a quick configuration of the analysis pipeline. 

## Processing pipeline
- File IO
  - [x] Multi-channel image handling pipeline
  - [x] Dask implementation 
- Preprocessing
  - [ ] Decovolution
  - [ ] Flat field correction
  - [ ] Denoising
  - [ ] Gamma correction
  - [x] Intensity normalization
  - [ ] Pixel size normalization
  - [x] Channel merge
- Segmentation
  
  Detection masks and cell centroids are expect to be in trackpy compatible format
  - [ ] Simple thresholding
  - [ ] Otsu thresholding
  - [x] StarDist
  - [ ] Cellpose
  - [ ] Morphological operations
  - [x] Connected components 
- Tracking
  - [x] Feature measurements
  - [ ] trackpy
  - [ ] TrackMate (pyImageJ integration, script based automation, with user provided centroids/segmentation masks)
  - [ ] btrack
- Contact Tracing
  - [ ] Number of contacts (same cell type/ different cell types)
  - [ ] Contact time measurements
  - [ ] Number of cell killing
  - [ ] Cell death event
  - [ ] Which cell(s) are interaction
  - [ ] Pandas/CSV export in trackpy format
  - [ ] Displacement, velocity
  - [ ] Analysis plots
 
The result Pandas table should looks like the following
| Id | Track | x | y | z | t | alive | vel_x | vel_z | vel_y | contact_same | contact_diff | contact_same_id | contacting_diff_id |
|:--:|:-----:|:-:|:-:|---|---|-------|-------|-------|-------|--------------|--------------|-----------------|--------------------|
|  0 |   1   |   |   |   |   |       |       |       |       |              |              |                 |                    |
|  1 |   2   |   |   |   |   |       |       |       |       |              |              |                 |                    |
|  2 |   1   |   |   |   |   |       |       |       |       |              |              |                 |                    |
|  3 |   1   |   |   |   |   |       |       |       |       |              |              |                 |                    |

## Environment Setup
Python virtual environment is high recommended. For quick setup, we recommend to use [mambaforge](https://github.com/conda-forge/miniforge#miniforge3) as a replacement of anaconda. Replace the `mamba` command if you choose to stay with `conda`.

Create virtual environment:
```bash
mamba create -n cyto python=3.10 -y
mamba activate cyto
```

### CUDA Acceleration
In some process we can accelerate the analysis process with CUDA GPU. To achieve CUDA version compatibility **Tensorflow 2.12.0** and **pyTorch 2.0.1** must be used together with **CUDA 11.8**. 

#### Dask CUDA
[Dask CUDA](https://github.com/rapidsai/dask-cuda) provides various utilities to improve deployment and management of Dask workers on CUDA-enabled systems.

We have to install through mamba/conda instead of pip for better version control.
```bash
mamba install -c rapidsai -c conda-forge -c nvidia dask-cuda cudatoolkit=11.8
```

#### Tensorflow
For Stardist segmentation we recommend to use TF2 verison. Follow the instruction for [TF 2.12 installation](https://www.tensorflow.org/install/pip).

#### pyTorch
For Cellpose segmentation we recommend to use pyTorch 2.0.. Follow the instruction for [pyTorch installation](https://pytorch.org/get-started/locally/).

### Required Packages

```bash
git clone git@github.com:bpi-oxford/Cytotoxicity-Pipeline.git
cd Cytotoxicity-Pipeline
pip install -r requirements.txt
```

## Usage

```bash
python cyto.py --pipeline <path-to-pipeline.yaml> -v
```

During runtime a Dask daskboard is created. Usually you can access with the address: http://localhost:8787/status, but check if the port is matches output on the line:
```log
2023-05-26 18:19:55,929 - distributed.scheduler - INFO -   dashboard at:  http://129.67.90.167:8787/status
```

### YAML Example
Check the pipeline YAML example in [./examples/pipeline.yaml](./examples/pipeline.yaml)

## Development Guide
For developers please follow the guide in [./doc/dev_guide.md](./doc/dev_guide.md).

## Authors
Jacky Ka Long Ko: [ka.ko@kennedy.ox.ac.uk](mailto:ka.ko@kennedy.ox.ac.uk)
