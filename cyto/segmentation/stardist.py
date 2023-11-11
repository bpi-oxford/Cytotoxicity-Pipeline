from stardist.models import StarDist2D
from typing import Any
from tqdm import tqdm
import dask.array as da
import tensorflow as tf
import numpy as np
from csbdeep.utils import normalize
import gc

class StarDist(object):
    def __init__(self,model_name="2D_versatile_fluo", prob_thresh=0.479071, nms_thresh=0.3, verbose=True) -> None:
        """
        Perform StarDist segmentation model

        Args:
            model_name (str): Registered models for StarDist2D
            prob_thresh (float): Probability threshold between 0 and 1
            nms_thresh (float): Non-maximum suppression threshold between 0 and 1
            verbose (bool): Turn on or off the processing printout
        """
        self.name = "StarDist"
        self.model_name = model_name
        self.verbose = verbose
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh

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
        label = np.zeros_like(image)
        for t in tqdm(range(image.shape[2])):
            img = image[:,:,t]
            if isinstance(img, da.Array):
                img = img.compute()
            img = normalize(img,pmin=0,pmax=100,axis=(0,1)) # convert from range [0,max] to [0,1]
            label_, _ = model.predict_instances(img,prob_thresh=self.prob_thresh,nms_thresh=self.nms_thresh)
            label[:,:,t] = label_
        label = label.astype(np.uint16)

        # collect memeory garbage manually
        # gc.collect()
        tf.keras.backend.clear_session()
        
        return {"image": image, "label": label}
