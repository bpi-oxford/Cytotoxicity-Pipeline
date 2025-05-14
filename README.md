<p align="center">
  <img src="./doc/assets/logo.png" alt="pyCyto Logo" width="150"/>
</p>

# Python Cytotoxicity Pipeline (pyCyto)

**pyCyto** is a flexible and extensible Python-based workflow package designed for analyzing microscopic cytotoxicity assays. It enables researchers and bioimage analysts to build reproducible analysis pipelines using simple YAML configuration files. Key features include support for multi-channel images, integration with Dask for parallel processing, and a modular design covering preprocessing, segmentation, tracking, and postprocessing analysis.

---

## 🔬 Cytotoxicity Assays and Cine Microscopy Analysis
pyCyto provides robust tools for analyzing key cellular behaviors in time-lapse microscopy. The core analysis features include:

- **Cell Positioning (Segmentation)**: Detection and localization of individual cells in each frame using classical or deep-learning-based image segmentation.

- **Temporal Tracking**: Linking cell positions across consecutive frames to track individual cells over time and reconstruct their trajectories.

- **Single-Cell Level Measurements**: For each tracked cell, pyCyto profiles:

  - **Fluorescent Intensity**: Quantification of channel-specific signal levels.

  - **Cell Alive Signal**: Estimation of cell viability based on death marker presence.

  - **Kinematics Measurement**: Analysis of cell movement, including velocity and displacement vectors.

  - **Cell Contacting Count**: Enumeration and logging of contact events between cells throughout the time course.

---

## 🧪 Extended Applications

Beyond standard cytotoxicity assays, **pyCyto** is also tailored for analyzing **cell-cell interaction processes**, especially in **pathogen–immune cell killing assays**. It enables tracking and quantifying interactions between cytotoxic immune cells (e.g., T cells, NK cells) and target/infected cells over time.

Other supported use cases include:

* **Vesicle-mediated killing assays**
* **General time-lapse microscopy of dynamic cell processes**
* Compatible with **fluorescent and non-fluorescent imaging modalities**

### 🧰 Assay Design Principles

* **One channel per cell type** (e.g., immune and target cells)
* **One marker channel** for cell death indicators (e.g., Propidium Iodide, Annexin V)

This design provides a flexible framework for multi-dimensional single-cell analysis across a variety of biological contexts.

---

## 📈 Processing Pipeline

The schematic of the cytotoxicity analysis pipeline can be found in [./doc/source/pipeline.md](./doc/source/pipeline.md).

**pyCyto** aims to automate the process of generating result tables from cytotoxicity microscopy images in the following format:

| Id | Track\_id | x | y | z | t | ch0\_signal | ch1\_signal | alive\_probability | vel\_x | vel\_y | vel\_z | contact | contact\_cell\_id | contact\_count |
| -- | --------- | - | - | - | - | ----------- | ----------- | ------------------ | ------ | ------ | ------ | ------- | ----------------- | -------------- |
| 0  | 1         |   |   |   |   |             |             |                    |        |        |        |         |                   |                |
| 1  | 2         |   |   |   |   |             |             |                    |        |        |        |         |                   |                |
| 2  | 1         |   |   |   |   |             |             |                    |        |        |        |         |                   |                |
| 3  | 1         |   |   |   |   |             |             |                    |        |        |        |         |                   |                |

---

## 📚 Documentation

Comprehensive documentation, including tutorials, API references, and detailed explanations, is available on the [documentation site](https://bpi-oxford.github.io/Cytotoxicity-Pipeline/index.html). The source files for the documentation, built with Sphinx, reside in the [`./doc`](./doc) directory.

---

## 🧪 Example Data

Example datasets for testing the pipeline can be found in the [`./data`](./data) directory. Due to large file size, data is managed by Git LFS. See the documentation for details on acquiring and using the example data.

---

## ✅ Test Cases

End-to-end test cases demonstrating pipeline usage with example data are located in the [`./tests`](./tests) directory. These can be used to verify installation and understand different configuration options.

---

## ⚙️ Environment Setup

### Native Python Runtime

Check [./doc/source/setup.md](./doc/source/setup.md) for setup instructions.

### Docker Runtime

For quick environment setup, use the Docker image:

```bash
docker build --pull --rm -f "Dockerfile" -t cytotoxicity-pipeline:latest "."
docker run --gpus all -u $(id -u):$(id -g) -v <path-to-data>:/data --rm -it -p 8787:8787/tcp cytotoxicity-pipeline:latest bash
```

---

## 📦 Package Installation

### Using PIP

```bash
pip install git+https://github.com/bpi-oxford/Cytotoxicity-Pipeline
```

### From Source

```bash
git clone git@github.com:bpi-oxford/Cytotoxicity-Pipeline.git
pip install -e .
```

---

## 🚀 Usage

```bash
# Single Node
cyto --pipeline <path-to-pipeline.yaml> -v

# Run from source
python main.py --pipeline <path-to-pipeline.yaml> -v
```

During runtime, a Dask dashboard is created. You can usually access it at: [http://localhost:8787/status](http://localhost:8787/status). Check the console output for the actual port:

```log
2023-05-26 18:19:55,929 - distributed.scheduler - INFO -   dashboard at:  http://129.67.90.167:8787/status
```

### YAML Configuration Example

See [./pipelines/pipeline.yaml](./pipelines/pipeline.yaml) for a complete example.

---

## 👩‍💻 Development Guide

For contributors and developers, please follow the guide in [./doc/source/dev\_guide.md](./doc/source/dev_guide.md).

---

## 📌 TODO

* GUI pipeline configuration
* SimpleITK and OpenCV function wrappers

---

## 👥 Authors

* Jacky Ka Long Ko: [ka.ko@kennedy.ox.ac.uk](mailto:ka.ko@kennedy.ox.ac.uk)
* Veronika Pfannenstill: [veronika.pfannenstill@stx.ox.ac.uk](mailto:veronika.pfannenstill@stx.ox.ac.uk)
* Samuel Alber: [salber@berkeley.edu](mailto:salber@berkeley.edu)
