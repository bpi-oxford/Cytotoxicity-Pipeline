from typing import Any

class ImageToLabel(object):
    def __init__(self,lp=5,up=95, verbose=True) -> None:
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