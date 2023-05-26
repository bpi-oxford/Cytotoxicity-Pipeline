from typing import Any
from tqdm import tqdm
from skimage import exposure
import dask.array as da
import numpy as np

class PercentileNormalization(object):
    def __init__(self,lp=5,up=95, verbose=True) -> None:
        """
        Perform percentile normalization across whole image data.

        Args:
            lp (float): Lower percentile
            up (float): Upper percentile
            verbose (bool): Turn on or off the processing printout
        """
        self.name = "PercentileNormalization"
        self.lp = lp
        self.up = up
        self.verbose = verbose

    def __call__(self, data) -> Any:
        image = data["image"]

        if self.verbose:
            tqdm.write("Percentile normalization: [{},{}]".format(self.lp,self.up))

        # normalization across time
        in_range = (da.percentile(image.flatten(),self.lp).compute()[0],da.percentile(image.flatten(),self.up).compute()[0])

        image = exposure.rescale_intensity(
            image,
            in_range=in_range,
            out_range=(0,np.iinfo(image.dtype).max)
        )

        return {"image": image}