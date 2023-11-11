from typing import Any
import numpy as np

class ImageToLabel(object):
    def __init__(self, verbose=True) -> None:
        """
        Convert image to label.

        Args:
            verbose (bool): Turn on or off the processing printout
        """
        self.name = "ImageToLabel"
        self.verbose = verbose

    def __call__(self, data) -> Any:
        image = data["image"]
        img_type = image.dtype

        return {"image": image, "label": image}
    
class ChannelMerge(object):
    def __init__(self, weights=None, verbose=False):
        """
        Temporal weighted merge of multiple images into one single image

        Args:
            weights (list): List of manual weight input, if None then will automatically computed
            verbose (bool): Turn on or off the processing printout
        """
        self.name = "ChannelMerge"
        self.weights = weights
        self.verbose = verbose

    def __call__(self, data) -> Any:
        images = data["images"] # note the difference between "image" and "images"

        # weighted combination
        weighted_combine = np.zeros_like(images[0])

        # weighted combination
        if self.weights == None:
            ch_means = [np.mean(image) for image in images]
            self.weights = [1/i for i in ch_means]

        weighted_combine = np.zeros_like(images[0])

        for ch in range(len(images)):
            weighted = self.weights[ch]/np.sum(self.weights)*images[ch]
            weighted_combine += weighted.astype(images[0].dtype)

        return {"output": weighted_combine}