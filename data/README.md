# Example Data

This directory is intended to store example datasets for running and testing the pyCyto pipeline.

**Note:** Due to the potential size of microscopy datasets, files in this directory (e.g., `.tif`, `.zarr`) are managed using **Git LFS (Large File Storage)**.

## Obtaining the Data

1.  **Install Git LFS:** If you haven't already, install Git LFS on your system. Follow the instructions at [https://git-lfs.github.com/](https://git-lfs.github.com/). You typically only need to run `git lfs install` once per user account.
2.  **Clone the Repository:** If you haven't cloned the repository yet, do so:
    ```bash
    git clone git@github.com:bpi-oxford/Cytotoxicity-Pipeline.git
    cd Cytotoxicity-Pipeline
    ```
3.  **Pull LFS Files:** After cloning, or if you've pulled updates and the LFS files weren't automatically downloaded, you can explicitly pull them:
    ```bash
    git lfs pull
    ```
    This command downloads the large files tracked by Git LFS.

If you encounter issues, ensure Git LFS is correctly installed and initialized (running `git lfs install --system` might be needed before cloning or pulling).
