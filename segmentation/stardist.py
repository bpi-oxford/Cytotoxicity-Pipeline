from stardist.models import StarDist2D
from typing import Any
from tqdm import tqdm
import dask.array as da
import tensorflow as tf

class StarDist(object):
    def __init__(self,model_name="2D_versatile_fluo", verbose=True) -> None:
        """
        Perform StarDist segmentation model

        Args:
            model_name (str): Registered models for StarDist2D
            verbose (bool): Turn on or off the processing printout
        """
        self.name = "StarDist"
        self.model_name = model_name
        self.verbose = verbose

    def __call__(self, data) -> Any:
        image = data["image"]

        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

        model = StarDist2D.from_pretrained(self.model_name)

        if self.verbose:
            tqdm.write("StarDist Segmentation 2D: {}".format(self.model_name))

        # perform stardist segmentation
        label = da.zeros_like(image)
        for t in tqdm(range(image.shape[2])):
            label[:,:,t], _ = model.predict_instances(image[:,:,t])

        return {"image": image, "label": label}
