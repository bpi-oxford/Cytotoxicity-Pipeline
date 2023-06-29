from typing import Any
import pandas as pd
from tqdm import tqdm
import os
from skimage import exposure
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class KernelDensityEstimation(object):
    def __init__(self, base_image=True, downscale_factor=1, gradient=False, verbose=True) -> None:
        """
        Perform kernel density estimation with given sparse dataframe
        
        Args:
            base_image (bool): Plot the KDE clusters with base image
            downscale_factor (float): Downscale resizing factor for the output KDE image (default=1)
            gradient (bool): Compute gradient of the KDE output (default: False)
            verbose (bool): Turn on or off the processing printout
        """
        self.base_image = base_image
        self.downscale_factor = downscale_factor
        self.gradient = gradient
        self.verbose = verbose
    
    def __call__(self, data) -> Any:
        image = data["image"]
        feature = data["feature"]
        START_FRAME = feature.frame.min()
        END_FRAME = feature.frame.max()

        kde_image = None

        for CUR_FRAME in tqdm(range(START_FRAME,END_FRAME+1)):
            frame = image[:,:,CUR_FRAME]
            sub = feature[feature.frame==CUR_FRAME]

            # scatter plot to show cell centroids
            fig, ax = plt.subplots(1,1,figsize=(5,5))
            plt.tight_layout(pad=0)

            ax.set_axis_off()
            if self.base_image:
                tail = 1
                pl, pu = np.percentile(frame, (tail, 100-tail))
                ax.imshow(exposure.rescale_intensity(frame, in_range=(pl, pu),out_range=(0,255)),cmap='gray')

            # density calculation using kde with grid search CV to optimize bandwidth
            params = {'bandwidth': np.logspace(-5,5,100), 'metric': ['euclidean']}
            grid = GridSearchCV(KernelDensity(), params)
            fit_coord = sub[['i','j']]

            grid.fit(fit_coord)

            tqdm.write("Best KDE bandwidth found: {0}".format(grid.best_estimator_.bandwidth))

            kde = grid.best_estimator_

            # for speed up only
            INTERVAL =  self.downscale_factor
            xgrid = np.arange(0,frame.shape[0])
            ygrid = np.arange(0,frame.shape[1])
            X,Y = np.meshgrid(xgrid[::INTERVAL],ygrid[::INTERVAL])
            xy = np.vstack([X.ravel(),Y.ravel()]).T
            Z = np.exp(kde.score_samples(xy))
            # Z = kde.score_samples(xy)
            Z = Z.reshape(X.shape)
            # Z /= Z.sum()
            Z /= Z.max()
            levels = np.linspace(0, Z.max(), 25)

            # cf = ax.contourf(X,Y,Z,cmap=plt.cm.rainbow,alpha=0.5,levels=levels)

            # gradient of the cluster density
            if self.gradient:
                lap = cv2.Laplacian(Z*255,cv2.CV_64F,ksize=3) 
                lap = np.uint8(np.absolute(lap))

            if kde_image is None:
                kde_image = np.expand_dims(Z, axis=-1)
            else:
                kde_image = np.concatenate([kde_image,np.expand_dims(Z,axis=-1)],axis=-1)

        return {"image": kde_image}