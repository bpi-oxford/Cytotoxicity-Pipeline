import numpy as np
from typing import Any
from tqdm import tqdm

def pixelwise_correlation(img1,img2, method = "MEAN",scale=1024e6):
    
    if method == "MEAN":
        # mean value
        img1_char_value = np.mean(img1)
        img2_char_value = np.mean(img2)

    else:
        # median value
        img1_char_value = np.median(img1.compute())
        img2_char_value = np.median(img2.compute())

    # variance
    img1_var = np.power(img1-img1_char_value,2)
    img2_var = np.power(img2-img2_char_value,2)

    # denumerator
    denum = np.sqrt(np.sum(img1_var)*np.sum(img2_var))

    # pixel correlation
    corr = (img1-img1_char_value)*(img2-img2_char_value)/denum

    # scaling and offset
    corr = scale*corr+(np.iinfo(np.uint16).max/2)

    return corr.compute().astype(np.uint16)

class CrossChannelCorrelation(object):
    def __init__(self, mode="SLICE", scale=1024e6, method = "MEAN", verbose=False):
        """
        Pixelwise Pearson correlation coefficients between two channels. Formulation of the correlation:
        $$\textup{corr}_i = \frac{(I_{0,i}-\bar{I}_0)(I_{1,i}-\bar{I}_1)}{\sum_i(I_{0,i}-\bar{I}_0)^2(I_{1,i}-\bar{I}_1)^2}$$

        Output dtype is uint16, that the correlation is remapped from float by the following:
        $$\textup{corr}_{out}= \textup{scale}\cdot\textup{corr}_{raw} + \frac{2^{16}-1}{2}$$
        
        Args:
            mode (str): Sample mean across SLICE or TIME (default = SLICE)
            scale (int): Output scaling (default = 1024e6)
            verbose (bool): Turn on or off the processing printout
            method (string): Choose characteristic value method among MEAN or MEDIAN (default = MEAN)
        """
        self.name = "CrossChannelCorrelation"
        assert mode in ["SLICE","TIME"] , "CrossChannelCorrelation mode only allow SLICE or TIME, {} given".format(mode)

        self.mode = mode
        self.scale=scale
        self.verbose = verbose
        self.method= method

    def __call__(self, data) -> Any:
        images = data["images"] # note the difference between "image" and "images"
        corr = np.zeros(images[0].shape,dtype=np.uint16)

        # cross channel correlation
        if self.mode == "SLICE":
            for t in tqdm(range(images[0].shape[2])):
                corr[:,:,t] = pixelwise_correlation(images[0][:,:,t], images[1][:,:,t], method = self.method, scale=self.scale)    
        else:
            corr = pixelwise_correlation(images[0],images[1], method = self.method,scale=self.scale)

        return {"output": corr}