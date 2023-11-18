# from aicsimageio import AICSImage
from cyto.postprocessing.graph import *
from cyto.utils.label_to_table import *
import dask_image.imread
import numpy as np
import pyclesperanto_prototype as cle
import pandas as pd
import pytest

def test_CrossCellContactMeasures():
    CANCER_IMAGE_PATH_PATTERN = "/mnt/Data/UTSE/2023_10_17_Nyeso1HCT116_1G4CD8_icam_FR10s_0p1mlperh/roi/register_denoise_gamma_channel_merged/cancer/*.tif"
    TCELL_IMAGE_PATH_PATTERN = "/mnt/Data/UTSE/2023_10_17_Nyeso1HCT116_1G4CD8_icam_FR10s_0p1mlperh/roi/register_denoise_gamma_channel_merged/tcell/*.tif"

    CANCER_LABEL_PATH_PATTERN = "/mnt/Data/UTSE/2023_10_17_Nyeso1HCT116_1G4CD8_icam_FR10s_0p1mlperh/roi/register_denoise_gamma_channel_merged_masks/cancer/*.tif"
    TCELL_LABEL_PATH_PATTERN = "/mnt/Data/UTSE/2023_10_17_Nyeso1HCT116_1G4CD8_icam_FR10s_0p1mlperh/roi/register_denoise_gamma_channel_merged_masks/tcell/*.tif"

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
    cccm = CrossCellContactMeasures(base_image=True,threads=2)
    res = cccm(data)

    # print(res)