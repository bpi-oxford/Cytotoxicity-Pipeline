# Tests

This directory contains tests for the pyCyto package.

## Structure

*   **Unit Tests:** Files like `test_*.py` typically contain unit tests focusing on specific functions or modules within the `cyto/` package. These are often run using `pytest`.
*   **Integration/End-to-End Tests:** This directory may also contain scripts or configurations for running more comprehensive tests, potentially involving full pipeline executions using data from the `/data` directory. These serve as test cases to verify overall functionality.

## Purpose

These tests serve to:

*   Ensure the correctness and reliability of individual code components (unit tests).
*   Verify the correct installation and functioning of the pipeline in integrated scenarios (end-to-end tests).
*   Demonstrate how to configure and run different analysis workflows using the YAML pipeline files.
*   Provide examples for users to adapt for their own experiments.

## Running Tests

Refer to the main project documentation or specific test scripts for instructions on how to execute these tests (e.g., using `pytest`). You might need to ensure the example data is available locally first for end-to-end tests.
