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

        weighted_combine = np.zeros_like(images[0])

        for ch in range(len(images)):
            weighted_combine += ch_means[ch]/np.sum(ch_means)*images[ch]
        weighted_combine.astype(images[0].dtype)

        return {"output": weighted_combine}