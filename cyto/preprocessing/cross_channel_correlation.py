import numpy as np
from typing import Any
from tqdm import tqdm

def pixelwise_correlation(img1,img2):
    # mean value
    img1_mean = np.mean(img1)
    img2_mean = np.mean(img2)

    # variance
    img1_var = np.power(img1-img1_mean,2)
    img2_var = np.power(img2-img2_mean,2)

    # denumerator
    denum = np.sqrt(np.sum(img1_var)*np.sum(img2_var))

    # pixel correlation
    corr = (img1-img1_mean)*(img2-img2_mean)/denum

    # scaling and offset
    print(np.min(corr).compute(),np.max(corr).compute())
    corr = 1024*corr*1e6+(np.iinfo(np.uint16).max/2)

    return corr.compute().astype(np.uint16)

class CrossChannelCorrelation(object):
    def __init__(self, mode="SLICE", verbose=False):
        """
        Pixelwise Pearson correlation coefficients between two channels

        Args:
            mode (str): Sample mean across SLICE or TIME (default = SLICE)
            verbose (bool): Turn on or off the processing printout
        """
        self.name = "CrossChannelCorrelation"
        assert mode in ["SLICE","TIME"] , "CrossChannelCorrelation mode only allow SLICE or TIME, {} given".format(mode)

        self.mode = mode
        self.verbose = verbose

    def __call__(self, data) -> Any:
        images = data["images"] # note the difference between "image" and "images"
        corr = np.zeros(images[0].shape,dtype=np.uint16)

        # cross channel correlation
        if self.mode == "SLICE":
            for t in tqdm(range(images[0].shape[2])):
                corr[:,:,t] = pixelwise_correlation(images[0][:,:,t], images[1][:,:,t])    
        else:
            corr = pixelwise_correlation(images[0],images[1])

        return {"output": corr}