# Cytotoxicity-Pipeline
A pipelined workflow for confocal cytotoxicity data.

Users may provide a YAML sript for a quick configuration of the analysis pipeline. 

## Processing pipeline
- File IO
  - [ ] Multi-channel image handling pipeline
  - [ ] Dask implementation 
- Preprocessing
  - [ ] Decovolution
  - [ ] Denoising
  - [ ] Gamma correction
  - [ ] Intensity normalization
  - [ ] Pixel size normalization
- Segmentation
  Detection masks and cell centroids are expect to be in trackpy compatible format
  - [ ] StarDist
  - [ ] Cellpose
- Tracking
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
 
The result Pandas table should looks like the following
| Id | Track | x | y | z | t | alive | vel_x | vel_z | vel_y | contact_same | contact_diff | contact_same_id | contacting_diff_id |
|:--:|:-----:|:-:|:-:|---|---|-------|-------|-------|-------|--------------|--------------|-----------------|--------------------|
|  0 |   1   |   |   |   |   |       |       |       |       |              |              |                 |                    |
|  1 |   2   |   |   |   |   |       |       |       |       |              |              |                 |                    |
|  2 |   1   |   |   |   |   |       |       |       |       |              |              |                 |                    |
|  2 |   1   |   |   |   |   |       |       |       |       |              |              |                 |                    |

## Environment Setup
Python virtual environment is high recommended. For quick setup, we recommend to use [mambaforge](https://github.com/conda-forge/miniforge#miniforge3) as a replacement of anaconda. Replace the `mamba` command if you choose to stay with `conda`.

Create virtual environment:
```bash
mamba create -n cyto python=3.10 -y
mamba activate cyto
```
Install necessary packages

```bash
git clone git@github.com:bpi-oxford/Cytotoxicity-Pipeline.git
cd Cytotoxicity-Pipeline
pip install -r requirements.txt
```

### Tensorflow/pyTroch Installation

### CUDA Acceleration
In some process we can accelerate the analysis process with CUDA GPU. To achieve so you need to install proper CUDA libraries:

```bash
mamba install
```

## Usage

```bash
python cyto.py --pipeline <path-to-pipeline.yaml> -v
```

### YAML Example
