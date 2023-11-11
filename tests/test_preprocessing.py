# from aicsimageio import AICSImage
from cyto.preprocessing.cross_channel_correlation import *

import os
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from aicsimageio.writers.ome_zarr_writer import OmeZarrWriter
import dask
import dask.array as da
import dask_image.imread
import pytest

def test_CrossChannelCorrelation():
    # test data directory
    #ch0_path = "/mnt/Data/UTSE/2023_10_17_Nyeso1HCT116_1G4CD8_icam_FR10s_0p1mlperh/register_denoise_gamma/ch0"
    ch1_path = "/mnt/Data/UTSE/2023_10_17_Nyeso1HCT116_1G4CD8_icam_FR10s_0p1mlperh/register_denoise_gamma/ch1"
    ch2_path = "/mnt/Data/UTSE/2023_10_17_Nyeso1HCT116_1G4CD8_icam_FR10s_0p1mlperh/register_denoise_gamma/ch2"

    ch0_file_pattern = os.path.join(ch2_path,"*.tif")
    ch1_file_pattern = os.path.join(ch1_path,"*.tif")

    images = {}
	
    # # Get an AICSImage object
    # imgs = []
    # imgs.append(AICSImage(ch0_path))
    # imgs.append(AICSImage(ch1_path))

    # # Pull only a specific chunk in-memory
    # for ch, img in enumerate(imgs):
    #     img_dask = img.get_image_dask_data("XYT")[slice(0,-1,1),slice(0,-1,1),slice(0,3,1)]
    #     images[ch] = img_dask.persist()

    # dask image loader for temporally separated images
    images[0] = dask_image.imread.imread(ch0_file_pattern).T[:,:,:3].persist()
    images[1] = dask_image.imread.imread(ch1_file_pattern).T[:,:,:3].persist()

    ccc = CrossChannelCorrelation(mode="SLICE")
    res = ccc({"images": images})

    # store the result
    output_dir =  "./output/preprocessing/CrossChannelCorrelation"
    os.makedirs(output_dir,exist_ok=True)
    output_file = os.path.join(output_dir,"T_cell_pi_cc.tif")
    print("Exporting result: {}".format(output_file))
    print(type(res["output"]))
    print(res["output"].dtype)
    OmeTiffWriter.save(res["output"].T, output_file, dim_order="TYX")

    # output_file = os.path.join(output_dir,"cancer_cell_pi_cc.zarr")
    # print("Exporting result: {}".format(output_file))
    # writer = OmeZarrWriter(output_file)
    # writer.write_image(res["output"].T,dimension_order="TYX")