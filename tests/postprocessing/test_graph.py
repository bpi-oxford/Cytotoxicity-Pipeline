# from aicsimageio import AICSImage
from cyto.postprocessing.graph import *
from cyto.utils.label_to_table import *
import dask_image.imread
import numpy as np
import pyclesperanto_prototype as cle
import pandas as pd
import pytest
import tifffile

@pytest.mark.mpi_skip
def test_CrossCellContactMeasures():
    CANCER_IMAGE_PATH_PATTERN = "/mnt/Data/UTSE/2023_10_17_Nyeso1HCT116_1G4CD8_icam_FR10s_0p1mlperh/roi/register_denoise_gamma_channel_merged/cancer/*.tif"
    TCELL_IMAGE_PATH_PATTERN = "/mnt/Data/UTSE/2023_10_17_Nyeso1HCT116_1G4CD8_icam_FR10s_0p1mlperh/roi/register_denoise_gamma_channel_merged/tcell/*.tif"

    CANCER_LABEL_PATH_PATTERN = "/mnt/Data/UTSE/2023_10_17_Nyeso1HCT116_1G4CD8_icam_FR10s_0p1mlperh/roi/register_denoise_gamma_channel_merged_masks/cancer/*.tif"
    TCELL_LABEL_PATH_PATTERN = "/mnt/Data/UTSE/2023_10_17_Nyeso1HCT116_1G4CD8_icam_FR10s_0p1mlperh/roi/register_denoise_gamma_channel_merged_masks/tcell/*.tif"

    OUTPUT_DIR = "/mnt/Data/UTSE/2023_10_17_Nyeso1HCT116_1G4CD8_icam_FR10s_0p1mlperh/test_res"

    # default chunk size: 1 image
    image_cancer = dask_image.imread.imread(CANCER_IMAGE_PATH_PATTERN)
    image_tcell = dask_image.imread.imread(TCELL_IMAGE_PATH_PATTERN)

    label_cancer = dask_image.imread.imread(CANCER_LABEL_PATH_PATTERN)
    label_tcell = dask_image.imread.imread(TCELL_LABEL_PATH_PATTERN)

    FRAMES = 5 # define number of frames to perform contact analysis

    # convert segmentation mask to trackpy style array
    features = {}

    # note that pyCyto works in XYT dim order but dask/tiff is in TYX, need to transpose the array
    # TODO: pixel spacing
    features["cancer"] = label_to_sparse(label=label_cancer[:FRAMES,:,:].T,image=image_cancer[:FRAMES,:,:].T,spacing=[1,1],channel_name="cancer")

    features["tcell"] = label_to_sparse(label=label_tcell[:FRAMES,:,:].T,image=image_tcell[:FRAMES,:,:].T,spacing=[1,1],channel_name="tcell")

    # Prepare data for pyCyto processing, beware of order of data input, here we calculate the T Cell to Cancer Cell contact
    data = {}
    data["images"] = [image_tcell[:FRAMES,:,:].T,image_cancer[:FRAMES,:,:].T]
    data["labels"] = [label_tcell[:FRAMES,:,:].T,label_cancer[:FRAMES,:,:].T]
    data["features"] = [features["tcell"], features["cancer"]]

    # Cross cell contact measurementsSimple plot of the contact time with the contact table output
    for i in range(0,-1,-1): # more than 1 thread not supported without MPI with pyclesperanto
        tqdm.write("Initiate with thread count = {}".format(2**i))
        start_time = time.time()
        cccm = CrossCellContactMeasures(base_image=True,threads=2**i,verbose=True)
        res = cccm(data)
        
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time = end_time - start_time

        # Print the elapsed time
        tqdm.write("Contact analysis elapsed time for thread count = {}: {:.4f}s".format(str(2**i),elapsed_time))

        print("saving result with thread count {}".format(2**i))
        out_dir = os.path.join(OUTPUT_DIR, "threads_{}".format(str(2**i)))
        os.makedirs(out_dir,exist_ok=True)
        
        os.makedirs(os.path.join(out_dir, "network_image"),exist_ok=True)

        for f in range(FRAMES):
            # network image
            tifffile.imwrite(os.path.join(out_dir,"network_image","im_{}.tif".format(str(f).zfill(5))), res["image"][:,:,f].T)
        res["feature"].to_csv(os.path.join(out_dir,"features.csv"))

@pytest.mark.mpi
def test_CrossCellContactMeasures_mpi():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank==0:
        CANCER_IMAGE_PATH_PATTERN = "/mnt/Data/UTSE/2023_10_17_Nyeso1HCT116_1G4CD8_icam_FR10s_0p1mlperh/roi/register_denoise_gamma_channel_merged/cancer/*.tif"
        TCELL_IMAGE_PATH_PATTERN = "/mnt/Data/UTSE/2023_10_17_Nyeso1HCT116_1G4CD8_icam_FR10s_0p1mlperh/roi/register_denoise_gamma_channel_merged/tcell/*.tif"

        CANCER_LABEL_PATH_PATTERN = "/mnt/Data/UTSE/2023_10_17_Nyeso1HCT116_1G4CD8_icam_FR10s_0p1mlperh/roi/register_denoise_gamma_channel_merged_masks/cancer/*.tif"
        TCELL_LABEL_PATH_PATTERN = "/mnt/Data/UTSE/2023_10_17_Nyeso1HCT116_1G4CD8_icam_FR10s_0p1mlperh/roi/register_denoise_gamma_channel_merged_masks/tcell/*.tif"

        OUTPUT_DIR = "/mnt/Data/UTSE/2023_10_17_Nyeso1HCT116_1G4CD8_icam_FR10s_0p1mlperh/test_res"

        # default chunk size: 1 image
        image_cancer = dask_image.imread.imread(CANCER_IMAGE_PATH_PATTERN)
        image_tcell = dask_image.imread.imread(TCELL_IMAGE_PATH_PATTERN)

        label_cancer = dask_image.imread.imread(CANCER_LABEL_PATH_PATTERN)
        label_tcell = dask_image.imread.imread(TCELL_LABEL_PATH_PATTERN)

        FRAMES = 5 # define number of frames to perform contact analysis

        # convert segmentation mask to trackpy style array
        features = {}

        # note that pyCyto works in XYT dim order but dask/tiff is in TYX, need to transpose the array
        # TODO: pixel spacing
        features["cancer"] = label_to_sparse(label=label_cancer[:FRAMES,:,:].T,image=image_cancer[:FRAMES,:,:].T,spacing=[1,1],channel_name="cancer")

        features["tcell"] = label_to_sparse(label=label_tcell[:FRAMES,:,:].T,image=image_tcell[:FRAMES,:,:].T,spacing=[1,1],channel_name="tcell")

        # Prepare data for pyCyto processing, beware of order of data input, here we calculate the T Cell to Cancer Cell contact
        data = {}
        data["images"] = [image_tcell[:FRAMES,:,:].T,image_cancer[:FRAMES,:,:].T]
        data["labels"] = [label_tcell[:FRAMES,:,:].T,label_cancer[:FRAMES,:,:].T]
        data["features"] = [features["tcell"], features["cancer"]]

        # Cross cell contact measurementsSimple plot of the contact time with the contact table output
        tqdm.write("Initiate with MPI backend")
        start_time = time.time()
        cccm = CrossCellContactMeasures(base_image=True,verbose=True,parallel_backend="MPI")
        res = cccm(data) # parse the input data to main process
        
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time = end_time - start_time

        # Print the elapsed time
        tqdm.write("Contact analysis elapsed time: {:.4f}s".format(elapsed_time))

        # print("saving results")
        # out_dir = os.path.join(OUTPUT_DIR, "mpi")
        # os.makedirs(os.path.join(out_dir, "network_image"),exist_ok=True)

        # for f in range(FRAMES):
        #     # network image
        #     tifffile.imwrite(os.path.join(out_dir,"network_image","im_{}.tif".format(str(f).zfill(5))), res["image"][:,:,f].T)
        # res["feature"].to_csv(os.path.join(out_dir,"features.csv"))
    else:
        cccm = CrossCellContactMeasures(base_image=True,verbose=True,parallel_backend="MPI")
        cccm.mpi_worker()