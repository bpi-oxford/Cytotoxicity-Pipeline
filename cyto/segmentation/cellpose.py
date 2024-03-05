# %%
import os
import numpy as np
from cellpose import models, io, core

from typing import Any
from tqdm import tqdm
import dask.array as da
from aicsimageio import AICSImage
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from aicsimageio.types import PhysicalPixelSizes
# import gc
import torch

class CellPose(object):
    def __init__(self, model_type='cyto', cellprob_thresh=-3, model_matching_thresh=10.0, gpu=True, channels =[0,0], batch_size = 16, diameter = 16.18, verbose=True) -> None:
        """
        Perform CellPose segmentation 2D

        Args:
            model_type (str): Registered models for CellPose
            cellprob_thresh (float): Probability threshold between -8 and 8
            model_matching_thresh (float): Non-maximum suppression threshold between 0 and 30
            gpu (bool): use GPU
            channels (list): [cytoplasm, nucleus] if NUCLEUS channel does not exist, set the second channel to 0, channels = [0,0] # IF YOU HAVE GRAYSCALE
            batch_size (int): used for GPU
            diameter (float): cell diameter
            verbose (bool): Turn on or off the processing printout
        """
        self.name = "CellPose"
        self.model_type = model_type
        self.verbose = verbose
        self.cellprob_thresh = cellprob_thresh
        self.model_matching_thresh = model_matching_thresh
        self.gpu = gpu
        self.channels = channels
        self.batch_size = batch_size
        self.diameter = diameter

        self.flow_threshold = 0 # (31.0 - self.model_matching_thresh)/10.0

    def __call__(self, data) -> Any:
        image = data["image"]

        use_GPU = core.use_gpu()    
        model = models.Cellpose(gpu=self.gpu, model_type=self.model_type)

        if self.verbose:
            tqdm.write("CellPose Segmentation 2D: {}\nGPU activated? {}".format(self.model_type,use_GPU))

        if isinstance(image, da.Array):
            image = image.compute()

        images_stack = []
        for t in range(image.shape[2]):
            images_stack.append(image[:,:,t])
        
        if self.verbose:
            tqdm.write("Running Cellpose segmentation..")

        masks, flows, styles, diams = model.eval(
            images_stack, 
            batch_size=self.batch_size,
            diameter=self.diameter, # in pixel
            cellprob_threshold=self.cellprob_thresh,
            flow_threshold=self.flow_threshold,
            channels=self.channels,
            stitch_threshold=0.0,
            do_3D=False,
            progress=True)

        # gc.collect()
        torch.cuda.empty_cache()

        masks = np.asarray(masks)
        label = np.transpose(masks,axes=[1,2,0])
        label = label.astype(np.uint16)

        if self.verbose:
            tqdm.write("Cellpose segmentation complete")

        return {"image": image, "label": label}
