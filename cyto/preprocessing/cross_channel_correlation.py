import numpy as np
from typing import Any

class CrossChannelCorrelation(object):
    def __init__(self, verbose):
        """
        Temporal weighted merge of multiple images into one single image

        Args:
            verbose (bool): Turn on or off the processing printout
        """
        self.name = "ChannelMerge"
        self.verbose = verbose

    def __call__(self, data) -> Any:
        images = data["images"] # note the difference between "image" and "images"

        # weighted combination
        ch_means = [np.mean(image) for image in images]
        weights = [1/i for i in ch_means]

        weighted_combine = np.zeros_like(images[0])

        for ch in range(len(images)):
            weighted = weights[ch]/np.sum(weights)*images[ch]
            weighted_combine += weighted.astype(images[0].dtype)

        return {"output": weighted_combine}