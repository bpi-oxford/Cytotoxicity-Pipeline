from typing import Any
import pandas as pd
from tqdm import tqdm
import os
import pyclesperanto_prototype as cle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import exposure
import networkx as nx

class CellTriangulation(object):
    def __init__(self, base_image=True, downscale_factor=1, gradient=False, verbose=True) -> None:
        """
        Perform kernel density estimation with given sparse dataframe
        
        Args:
            base_image (bool): Plot the cell networks with base image
            downscale_factor (float): Downscale resizing factor for the output network image (default=1)
            verbose (bool): Turn on or off the processing printout
        """
        self.base_image = base_image
        self.downscale_factor = downscale_factor
        self.verbose = verbose
    
    def __call__(self, data) -> Any:
        image = data["image"]
        feature = data["feature"]
        START_FRAME = feature.frame.min()
        END_FRAME = feature.frame.max()

        graph = []
        network_image = None

        for CUR_FRAME in tqdm(range(START_FRAME,END_FRAME+1)):
            frame = image[:,:,CUR_FRAME]
            tail = 1
            pl, pu = np.percentile(frame, (tail, 100-tail))

            centroids = feature[feature.frame==CUR_FRAME][['i','j']].T.to_numpy()

            # generate distance matrix
            distance_matrix = cle.generate_distance_matrix(centroids, centroids)

            # grid search for the smallest distance to generate one connected network
            pbar = tqdm(range(50,10001,10))
            for i in pbar:
                pbar.set_description(str(i))
                connection_matrix_se = cle.smaller_or_equal_constant(distance_matrix, constant=i)
                connection_matrix_lq = cle.greater_or_equal_constant(distance_matrix, constant=1)
                connection_matrix = cle.multiply_images(connection_matrix_se, connection_matrix_lq)

                mesh = cle.create_like(frame)
                cle.touch_matrix_to_mesh(centroids, connection_matrix, mesh)

                networkx_graph = cle.to_networkx(connection_matrix, centroids)

                if nx.is_connected(networkx_graph):
                    print("Cell network is connected with minimum distance {}".format(i))
                    graph.append(networkx_graph)
                    break

            fig, ax = plt.subplots(1,1,figsize=(5,5))
            plt.tight_layout(pad=0)
            ax.set_axis_off()

            mesh_device = cle.create_like(frame)
            cle.touch_matrix_to_mesh(centroids, connection_matrix, mesh_device)
            # if self.base_image:
            #     ax.imshow(exposure.rescale_intensity(frame, in_range=(pl, pu),out_range=(0,255)),cmap='gray')
            # cle.imshow(mesh_device, labels=True, plot=ax, alpha=0.7)

            mesh_host = cle.pull(mesh_device)

            # memory clean up
            mesh_device.data.release()
            if network_image is None:
                network_image = np.expand_dims(mesh_host, axis=-1)
            else:
                network_image = np.concatenate([network_image,np.expand_dims(mesh_host,axis=-1)],axis=-1)

        return {"image": network_image, "network": graph}