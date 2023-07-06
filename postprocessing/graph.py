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
import SimpleITK as sitk

class CellTriangulation(object):
    def __init__(self, base_image=True, verbose=True) -> None:
        """
        Perform cell centroid triangulation
        
        Args:
            base_image (bool): Plot the cell networks with base image
            verbose (bool): Turn on or off the processing printout
        """
        self.base_image = base_image
        self.verbose = verbose
    
    def __call__(self, data) -> Any:
        image = data["image"]
        feature = data["feature"]
        START_FRAME = feature.frame.min()
        END_FRAME = feature.frame.max()

        graph = []
        network_image = None

        for CUR_FRAME in tqdm(range(START_FRAME,END_FRAME+1)):
            # TODO: check device memory usage
            print("Frame: ", CUR_FRAME)
            os.system("nvidia-smi --query-gpu=memory.used --format=csv")
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

            mesh_device = cle.create_like(frame)
            cle.touch_matrix_to_mesh(centroids, connection_matrix, mesh_device)
            mesh_host = cle.pull(mesh_device)

            # memory clean up
            mesh_device.data.release()
            if network_image is None:
                network_image = np.expand_dims(mesh_host, axis=-1)
            else:
                network_image = np.concatenate([network_image,np.expand_dims(mesh_host,axis=-1)],axis=-1)

        return {"image": network_image, "network": graph}

def label_centroids_to_pointlist_sitk(label, cell_type=""):
    labels_centroid = {
        "label": [],
        "x": [],
        "y": [],
        "bbox_xstart": [],
        "bbox_ystart": [],
        "bbox_xsize": [],
        "bbox_ysize": [],
        "cell_type": []
    }

    labelStat = sitk.LabelShapeStatisticsImageFilter()
    labelStat.Execute(label)
    for l in tqdm(labelStat.GetLabels()):
        labels_centroid["label"].append(l)
        labels_centroid["x"].append(label.TransformPhysicalPointToContinuousIndex(labelStat.GetCentroid(l))[0])
        labels_centroid["y"].append(label.TransformPhysicalPointToContinuousIndex(labelStat.GetCentroid(l))[1])
        bbox_start = label.TransformPhysicalPointToContinuousIndex((labelStat.GetBoundingBox(l)[0],labelStat.GetBoundingBox(l)[1]))
        bbox_end = label.TransformPhysicalPointToContinuousIndex((
                labelStat.GetBoundingBox(l)[0] + labelStat.GetBoundingBox(l)[2],
                labelStat.GetBoundingBox(l)[1] + labelStat.GetBoundingBox(l)[3],
            ))
        labels_centroid["bbox_xstart"].append(bbox_start[0])
        labels_centroid["bbox_ystart"].append(bbox_start[1])
        labels_centroid["bbox_xsize"].append(bbox_end[0] - bbox_start[0])
        labels_centroid["bbox_ysize"].append(bbox_end[1] - bbox_start[1])
        labels_centroid["cell_type"].append(cell_type)

    return pd.DataFrame.from_dict(labels_centroid)

class CrossCellContactMeasures(object):
    def __init__(self, base_image=True, verbose=True) -> None:
        """
        Perform cross cell type contact measurements
        
        Args:
            base_image (bool): Plot the cell networks with base image
            verbose (bool): Turn on or off the processing printout
        """
        self.base_image = base_image
        self.verbose = verbose
    
    def __call__(self, data) -> Any:
        images = data["images"]
        labels = data["labels"]
        features = data["features"]
        assert len(images) == 2, "Input images must be 2"
        assert len(features) == 2, "Input features must be 2"
        START_FRAME = features[0].frame.min()
        END_FRAME = features[0].frame.max()

        graph = []
        network_image = None

        for CUR_FRAME in tqdm(range(START_FRAME,END_FRAME+1)):
            print("Frame: ", CUR_FRAME)
            os.system("nvidia-smi --query-gpu=memory.used --format=csv")
            # TODO: check device memory allocation and release
            tail = 1

            c = []
            c_count = []
            for i, label in enumerate(labels):
                # centroids = label_centroids_to_pointlist_sitk(sitk.GetImageFromArray(label[:,:,CUR_FRAME]))[["x","y"]].to_numpy().T

                centroids = features[i][features[i].frame==CUR_FRAME][["i","j"]].to_numpy().T
                tqdm.write("Cell type {} count: {}".format(i,centroids.shape[1]))

                c.append(centroids)
                c_count.append(centroids.shape[1])

            # generate distance matrix
            distance_matrix = cle.generate_distance_matrix(c[0], c[1])

            print(labels[0][:,:,CUR_FRAME].shape)

            statFilter = sitk.LabelShapeStatisticsImageFilter()
            statFilter.Execute(sitk.GetImageFromArray(labels[1][:,:,CUR_FRAME]))

            relabelFilter = sitk.RelabelComponentImageFilter()
            labels_ = relabelFilter.Execute(sitk.GetImageFromArray(labels[0][:,:,CUR_FRAME]))
            labels[0][:,:,CUR_FRAME] = sitk.GetArrayFromImage(labels_)

            labels_ = relabelFilter.Execute(sitk.GetImageFromArray(labels[1][:,:,CUR_FRAME]))
            labels[1][:,:,CUR_FRAME] = sitk.GetArrayFromImage(labels_)

            statFilter = sitk.LabelShapeStatisticsImageFilter()
            statFilter.Execute(sitk.GetImageFromArray(labels[1][:,:,CUR_FRAME]))

            overlap_matrix = cle.generate_binary_overlap_matrix(labels[0][:,:,CUR_FRAME], labels[1][:,:,CUR_FRAME])

            masked_distance_matrix = cle.multiply_images(overlap_matrix,distance_matrix)

            pointlist = np.concatenate(c,axis=1)
            distance_matrix_pivot = np.zeros((c_count[0]+c_count[1]+1,c_count[0]+c_count[1]+1))
            distance_matrix_pivot[(c[0].shape[1]+1):,1:(c[0].shape[1]+1)] = masked_distance_matrix[1:,1:]
            distance_matrix_pivot[1:(c[0].shape[1]+1),(c[0].shape[1]+1):] = masked_distance_matrix[1:,1:].T

            distance_matrix_pivot = cle.push(distance_matrix_pivot)
            distance_mesh_device = cle.create_labels_like(labels[0][:,:,CUR_FRAME])
            cle.touch_matrix_to_mesh(pointlist, distance_matrix_pivot,distance_mesh_device)

            networkx_graph_two_cell_types_overlap = cle.to_networkx(distance_matrix_pivot, pointlist)
            graph.append(networkx_graph_two_cell_types_overlap)
            distance_mesh_host = cle.pull(distance_mesh_device)

            # memory clean up

            distance_mesh_device.data.release()

            if network_image is None:
                network_image = np.expand_dims(distance_mesh_host, axis=-1)
            else:
                network_image = np.concatenate([network_image,np.expand_dims(distance_mesh_host,axis=-1)],axis=-1)

            # use the masked_distance_matrix to combine with features table

        return {"image": network_image, "network": graph}