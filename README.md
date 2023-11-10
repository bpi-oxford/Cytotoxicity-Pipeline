# Cytotoxicity-Pipeline (pyCyto)
A pipelined workflow package for microscopic cytotoxicity analysis, intentionally designed for high extension flexibility and experiment repeatability.

Users may provide a YAML script for a quick configuration of the analysis pipeline. 

## Processing pipeline
- File IO
  - [x] Multi-channel image handling pipeline
  - [x] Dask implementation 
  - [ ] OME ZARR IO

- Preprocessing
  - [ ] Decovolution
  - [ ] Flat field correction
  - [ ] Denoising
  - [ ] Gamma correction
  - [x] Intensity normalization
  - [ ] Pixel size normalization
  - [x] Channel merge
  - [ ] Cross channel pixel correlation

- Segmentation
  
  Detection masks and cell centroids are expect to be in trackpy compatible format
  - [ ] Simple thresholding
  - [ ] Otsu thresholding
  - [x] StarDist
  - [x] Cellpose
  - [ ] Morphological operations
  - [x] Connected components 
- Tracking
  - [x] Feature measurements
  - [ ] trackpy
  - [x] TrackMate (pyImageJ integration, script based automation, with user provided centroids/segmentation masks)
  - [ ] btrack
  - [ ] ultrack
- Contact Tracing
  - [ ] Number of contacts (same cell type/ different cell types)
  - [ ] Contact time measurements
  - [ ] Number of cell killing
  - [ ] Cell death event
  - [ ] Which cell(s) are interaction
  - [ ] Pandas/CSV export in trackpy format
  - [ ] Displacement, velocity
  - [ ] Analysis plots
 
The result Pandas table should take the following format:
| Id | Track | x | y | z | t | alive | vel_x | vel_z | vel_y | contact_same | contact_diff | contact_same_id | contacting_diff_id |
|:--:|:-----:|:-:|:-:|---|---|-------|-------|-------|-------|--------------|--------------|-----------------|--------------------|
|  0 |   1   |   |   |   |   |       |       |       |       |              |              |                 |                    |
|  1 |   2   |   |   |   |   |       |       |       |       |              |              |                 |                    |
|  2 |   1   |   |   |   |   |       |       |       |       |              |              |                 |                    |
|  3 |   1   |   |   |   |   |       |       |       |       |              |              |                 |                    |

## Environment Setup
### Native Python Runtime
Check [./doc/setup.md](./doc/setup.md) for the setup instruction.

### Docker Runtime
For quick environment setup use the docker image:
```bash
docker build --pull --rm -f "Dockerfile" -t cytotoxicity-pipeline:latest "."
docker run --gpus all -u $(id -u):$(id -g) -v <path-to-data>:/data --rm -it -p 8787:8787/tcp cytotoxicity-pipeline:latest bash
```

## Package Installation
### PIP
```bash
pip install git+https://github.com/bpi-oxford/Cytotoxicity-Pipeline
```

### From Source
```bash
git clone git@github.com:bpi-oxford/Cytotoxicity-Pipeline.git
pip install -e .
```

## Usage
```bash
# Single Node
cyto --pipeline <path-to-pipeline.yaml> -v

# Run from source
python main.py --pipeline <path-to-pipeline.yaml> -v
```

During runtime a Dask daskboard is created. Usually you can access with the address: http://localhost:8787/status, but check if the port is matches output on the line:
```log
2023-05-26 18:19:55,929 - distributed.scheduler - INFO -   dashboard at:  http://129.67.90.167:8787/status
```

### YAML Example
Check the pipeline YAML example in [./pipelines/pipeline.yaml](./pipelines/pipeline.yaml)

## Development Guide
For developers please follow the guide in [./doc/dev_guide.md](./doc/dev_guide.md).

## TODO
- GUI pipeline configuration
- SLURM distributed run

## Authors
- Jacky Ka Long Ko: [ka.ko@kennedy.ox.ac.uk](mailto:ka.ko@kennedy.ox.ac.uk)
- Veronika Pfannenstill: [veronika.pfannenstill@stx.ox.ac.uk](mailto:veronika.pfannenstill@stx.ox.ac.uk)
- Samuel Alber: [salber@berkeley.edu](mailto:salber@berkeley.edu)