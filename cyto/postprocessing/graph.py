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
import multiprocessing
import dask.array as da

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
    def __init__(self, base_image=True, threads=1, verbose=True) -> None:
        """
        Perform cross cell type contact measurements
        
        Args:
            base_image (bool): Plot the cell networks with base image
            threads (int): Number of threads for parallel processing (default=1).
            verbose (bool): Turn on or off the processing printout
        """
        self.base_image = base_image
        self.threads = threads
        self.verbose = verbose
    
    def run_single_frame(label_0, label_1, centroids_0,centroids_1, features_0, features_1, frame):
        """
        Single frame process run

        Args:
            label_0 (arr): Numpy array label of first cell type
            label_1 (arr): Numpy array label of second cell type
            cetnroids_0 (arr): Numpy array of centroid coordinates corresponding to label_0, in pixel space ij.
            centroids_1 (arr): Numpy array of centroid coordinates corresponding to label_1, in pixel space ij.
            features_0 (Dataframe): Dataframe of spares cell info of first cell type
            features_1 (Dataframe): Dataframe of spares cell info of second cell type
            frame (int): Frame number to async parallel processing positioning.
        """
        # get number of centroids in each cell type
        c_count = [len(centroids_0),len(centroids_1)]

        # generate distance matrix on gpu
        distance_matrix_device = cle.generate_distance_matrix(centroids_0, centroids_1)

        # relabel the input
        relabelFilter = sitk.RelabelComponentImageFilter()
        label_sitk_0 = relabelFilter.Execute(sitk.GetImageFromArray(label_0))
        label_0 = sitk.GetArrayFromImage(label_sitk_0)

        label_sitk_1 = relabelFilter.Execute(sitk.GetImageFromArray(label_1))
        label_1 = sitk.GetArrayFromImage(label_sitk_1)

        # cell shape measurement by SITK
        statFilter_0 = sitk.LabelShapeStatisticsImageFilter()
        statFilter_0.Execute(label_sitk_0)

        statFilter_1 = sitk.LabelShapeStatisticsImageFilter()
        statFilter_1.Execute(label_sitk_1)

        overlap_matrix_device = cle.generate_binary_overlap_matrix(label_0, label_1)

        masked_distance_matrix = cle.multiply_images(overlap_matrix_device,distance_matrix_device)

        pointlist = np.concatenate([centroids_0,centroids_1],axis=1)
        distance_matrix_pivot = np.zeros((c_count[0]+c_count[1]+1,c_count[0]+c_count[1]+1))
        distance_matrix_pivot[(centroids_0.shape[1]+1):,1:(centroids_1.shape[1]+1)] = masked_distance_matrix[1:,1:]
        distance_matrix_pivot[1:(centroids_0.shape[1]+1),(centroids_0.shape[1]+1):] = masked_distance_matrix[1:,1:].T

        distance_matrix_pivot = cle.push(distance_matrix_pivot)
        distance_mesh_device = cle.create_labels_like(label_0)
        cle.touch_matrix_to_mesh(pointlist, distance_matrix_pivot,distance_mesh_device)

        networkx_graph_two_cell_types_overlap = cle.to_networkx(distance_matrix_pivot, pointlist)
        graph = networkx_graph_two_cell_types_overlap # networkx graph

        # pulling data from device to host
        distance_matrix_host = cle.pull(distance_matrix_device)
        overlap_matrix_host = cle.pull(overlap_matrix_device)
        distance_mesh_host = cle.pull(distance_mesh_device)

        # memory clean up
        distance_matrix_device.data.release()
        overlap_matrix_device.data.release()
        distance_mesh_device.data.release()
        del distance_matrix_device
        del overlap_matrix_device
        del distance_mesh_device

        # use the masked_distance_matrix to combine with features table
        f_0 = features_0[features_0.frame==frame]

        contact = []
        contact_label = []
        closest_cell_dist = []

        for i in range(1,len(f_0.index)+1):
            overlap = overlap_matrix_host[1:,i]
            dist = distance_matrix_host[1:,i]

            cell_label_offset = len(features_1[features_1.frame<frame].index)

            contact.append(True) if np.sum(overlap)>0 else contact.append(False)
            contact_label.append(np.where(overlap == 1)[0]+cell_label_offset+1) # cell label starts from 1 so need to offset extra 1
            closest_cell_dist.append(np.min(dist))

        return {"graph":graph, "network_image": distance_mesh_host, "frame": frame, "contact": contact, "contact_label": contact_label, "closest_cell_dist": closest_cell_dist}

    def __call__(self, data) -> Any:
        images = data["images"]
        labels = data["labels"]
        features = data["features"]
        assert len(images) == 2, "Input images must be 2"
        assert len(features) == 2, "Input features must be 2"
        START_FRAME = features[0].frame.min()
        END_FRAME = features[0].frame.max()
        features_out = features[0].copy()

        graph = []
        network_image = None
        contact = []
        contact_label = []
        closest_cell_dist = []

        if self.threads <= 1:
            pbar = tqdm(range(START_FRAME,END_FRAME+1),desc="Cross cell contact measurements (single)")
            for CUR_FRAME in pbar:
                c = []
                c_count = []
                for i, label in enumerate(labels):
                    centroids = features[i][features[i].frame==CUR_FRAME][["i","j"]].to_numpy().T
                    c.append(centroids)
                    c_count.append(centroids.shape[1])

                # generate distance matrix
                distance_matrix_device = cle.generate_distance_matrix(c[0], c[1])

                relabelFilter = sitk.RelabelComponentImageFilter()
                labels_ = relabelFilter.Execute(sitk.GetImageFromArray(labels[0][:,:,CUR_FRAME]))
                labels[0][:,:,CUR_FRAME] = sitk.GetArrayFromImage(labels_)

                labels_ = relabelFilter.Execute(sitk.GetImageFromArray(labels[1][:,:,CUR_FRAME]))
                labels[1][:,:,CUR_FRAME] = sitk.GetArrayFromImage(labels_)

                statFilter = sitk.LabelShapeStatisticsImageFilter()
                statFilter.Execute(sitk.GetImageFromArray(labels[1][:,:,CUR_FRAME]))

                statFilter = sitk.LabelShapeStatisticsImageFilter()
                statFilter.Execute(sitk.GetImageFromArray(labels[1][:,:,CUR_FRAME]))

                overlap_matrix_device = cle.generate_binary_overlap_matrix(labels[0][:,:,CUR_FRAME], labels[1][:,:,CUR_FRAME])

                masked_distance_matrix = cle.multiply_images(overlap_matrix_device,distance_matrix_device)

                pointlist = np.concatenate(c,axis=1)
                distance_matrix_pivot = np.zeros((c_count[0]+c_count[1]+1,c_count[0]+c_count[1]+1))
                distance_matrix_pivot[(c[0].shape[1]+1):,1:(c[0].shape[1]+1)] = masked_distance_matrix[1:,1:]
                distance_matrix_pivot[1:(c[0].shape[1]+1),(c[0].shape[1]+1):] = masked_distance_matrix[1:,1:].T

                distance_matrix_pivot = cle.push(distance_matrix_pivot)
                distance_mesh_device = cle.create_labels_like(labels[0][:,:,CUR_FRAME])
                cle.touch_matrix_to_mesh(pointlist, distance_matrix_pivot,distance_mesh_device)

                networkx_graph_two_cell_types_overlap = cle.to_networkx(distance_matrix_pivot, pointlist)
                # warning: may have offset if START_FRAME not begin from zero, known error
                graph.append(networkx_graph_two_cell_types_overlap)

                # pulling data from device to host
                distance_matrix_host = cle.pull(distance_matrix_device)
                overlap_matrix_host = cle.pull(overlap_matrix_device)
                distance_mesh_host = cle.pull(distance_mesh_device)

                # memory clean up
                distance_matrix_device.data.release()
                overlap_matrix_device.data.release()
                distance_mesh_device.data.release()
                del distance_matrix_device
                del overlap_matrix_device
                del distance_mesh_device

                if network_image is None:
                    network_image = np.expand_dims(distance_mesh_host, axis=-1)
                else:
                    network_image = np.concatenate([network_image,np.expand_dims(distance_mesh_host,axis=-1)],axis=-1)

                # use the masked_distance_matrix to combine with features table
                f_0 = features[0][features[0].frame==CUR_FRAME]
                
                for i in range(1,len(f_0.index)+1):
                    overlap = overlap_matrix_host[1:,i]
                    dist = distance_matrix_host[1:,i]

                    cell_label_offset = len(features[1][features[1].frame<CUR_FRAME].index)

                    contact.append(True) if np.sum(overlap)>0 else contact.append(False)
                    contact_label.append(np.where(overlap == 1)[0]+cell_label_offset+1) # cell label starts from 1 so need to offset extra 1
                    closest_cell_dist.append(np.min(dist))
        else:
            # create multiprocessing pool
            pool = multiprocessing.Pool(processes=self.threads)

            pbar = tqdm(range(START_FRAME,END_FRAME+1),desc="Cross cell contact measurements (parallel)")

            network_image = da.zeros_like(labels[0])

            # initiate dicts for unsorted results
            graph_ = {}
            contact_ = {}
            contact_label_ = {}
            closest_cell_dist_ = {}

            for CUR_FRAME in pbar:
                c = []
                c_count = []
                labels_np = []
                # loop over the two label images
                for i, label in enumerate(labels):
                    centroids = features[i][features[i].frame==CUR_FRAME][["i","j"]].to_numpy().T
                    c.append(centroids)
                    c_count.append(centroids.shape[1])
                    labels_np.append(label[:,:,CUR_FRAME].compute())

                def callback(res):
                    pbar.update(1)

                    frame = res["frame"]

                    # network image
                    network_image_ = res["network_image"]
                    network_image[:,:,frame] = network_image_

                    # graph
                    graph_[frame: res["graph"]]

                    # contact
                    contact_[frame: res["contact"]]

                    # contact label
                    contact_label_[frame: res["contact_label"]]

                    # closest cell dist
                    closest_cell_dist_[frame: res["closest_cell_dist"]]

                def err_callback(err):
                    print(err)

                pool.apply_async(self.run_single_frame,(labels_np[0],labels_np[1],c[0],c[1],features[0],features[1],CUR_FRAME,),callback=callback,error_callback=err_callback)
            
            # close pool and wait to finish
            pool.close()
            pool.join()

            # reorder unsorted results
            graph = dict(sorted(graph_.items())).values()
            for c in dict(sorted(contact_.items())).values():
                contact.extend(c)
            for c in dict(sorted(contact_label_)).values():
                contact_label.extend(c)
            for c in dict(sorted(closest_cell_dist_)).values():
                closest_cell_dist.extend(c)

        features_out["contact"] = contact
        features_out["contacting cell labels"] = contact_label
        features_out["closest cell dist"] = closest_cell_dist

        return {"image": network_image, "feature": features_out, "network": graph}