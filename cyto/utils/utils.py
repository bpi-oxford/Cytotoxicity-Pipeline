from typing import Any
import numpy as np
import pyopencl as cl

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

def check_gpu_memory():
    # Get list of platforms (e.g., NVIDIA, AMD, Intel)
    platforms = cl.get_platforms()

    for platform in platforms:
        print(f"Platform: {platform.name}")
        # Get list of devices (e.g., GPUs, CPUs) for the current platform
        devices = platform.get_devices()

        for device in devices:
            if device.type == cl.device_type.GPU:
                # Print device name and other details
                print(f"Device: {device.name}")
                print(f"Global Memory Size: {device.global_mem_size / (1024 ** 2)} MB")
                print(f"Max Allocable Memory: {device.max_mem_alloc_size / (1024 ** 2)} MB")
                print(f"Local Memory Size: {device.local_mem_size / 1024} KB")
                print(f"Available: Yes")
            else:
                print(f"Device: {device.name} (Not a GPU)")