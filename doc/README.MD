# Project Documentation

This directory contains the source files for the project documentation, built using Sphinx.

## Local Development

### Prerequisites

Ensure you have [Mamba](https://github.com/conda-forge/miniforge) (or Conda) installed. You will also need to install the project dependencies from the root directory of the project. It's recommended to create a dedicated Mamba environment.

1.  **Create and activate a Mamba environment:**
    ```bash
    mamba create -n cyto python=3.10
    mamba activate cyto
    ```

2.  **Install dependencies:**
    Navigate to the **root directory** of this project (the one containing `requirements.txt`).
    ```bash
    pip install -r requirements.txt 
    ```

### Building the Documentation

To build the HTML documentation locally:

1.  Ensure your Mamba environment with the dependencies is activated.
2.  Navigate to the `doc` directory (this directory).
3.  Run the make command:
    ```bash
    cd ./source
    make html
    ```

This command will generate the HTML files in the `doc/_build/html` directory.

### Serving the Documentation Locally

After building the documentation, you can serve it locally to view it in your browser.

1.  Navigate to the directory where the HTML files were generated:
    ```bash
    cd _build/html  # This path is relative to the 'doc' directory
    ```

2.  Start a simple HTTP server using Python (which should be available in your Mamba environment):
    ```bash
    python -m http.server
    ```
    By default, this will serve the documentation at `http://localhost:8000`. Open this URL in your web browser to view the documentation.

    If port 8000 is in use, you can specify a different port:
    ```bash
    python -m http.server 8080 
    # This will serve on http://localhost:8080
    ```

## Automated Documentation Deployment

This project uses GitHub Actions to automatically build and deploy the documentation to GitHub Pages.

-   **Trigger**: The workflow is triggered on every push to the `main` branch.
-   **Process**:
    1.  The action checks out the latest code.
    2.  It sets up a Python environment (version 3.10).
    3.  Project dependencies (from `requirements.txt` in the root directory, excluding `mpi4py` if problematic for the build environment) are installed.
    4.  The Sphinx documentation is built using the command `make -C doc html`.
    5.  The generated HTML files from `doc/build/html` are then deployed to the `gh-pages` branch of this repository.
-   **Hosting**: The documentation is hosted on the corresponding [GitHub Pages](https://bpi-oxford.github.io/Cytotoxicity-Pipeline/index.html).

The workflow configuration can be found in `.github/workflows/docs.yml`.
