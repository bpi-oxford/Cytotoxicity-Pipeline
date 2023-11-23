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
from pathos.threading import ThreadPool
import time
import pickle
from datetime import datetime

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
    # MPI tags
    MPI_TAGS = {
        "TASK": 0,
        "RESULT": 1,
    }

    def __init__(self, threads=1, parallel_backend="NATIVE", verbose=True) -> None:
        """
        Perform cross cell type contact measurements
        
        Args:
            threads (int): Number of threads for parallel processing (default=1). Setting the thread number to high value may overflow system memory. Will be overrided by MPI rank size if use MPI backend.
            parallel_backend (str): Parallelization backend, note pyclespranto only work with MPI at this moment (default = "NATIVE")["MPI","NATIVE","PATHOS"]
            verbose (bool): Turn on or off the processing printout
        """
        self.threads = threads
        self.parallel_backend = parallel_backend
        self.verbose = verbose
        self.lock = multiprocessing.Lock()

    def run_single_frame(self,label_0, label_1, centroids_0,centroids_1, features_0, features_1,frame):
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
        if self.verbose:
            tqdm.write("frame {} thread started".format(frame))

        start_time_0 = time.time()

        # get number of centroids in each cell type
        c_count = [centroids_0.shape[1],centroids_1.shape[1]]
        if self.verbose:
            tqdm.write("Distance matrix scale: {} x {}".format(c_count[0], c_count[1]))

        # generate distance matrix on gpu
        start_time = time.time()
        distance_matrix_device = cle.generate_distance_matrix(centroids_0, centroids_1)

        if self.verbose:
            end_time = time.time()
            # Calculate elapsed time
            elapsed_time = end_time - start_time

            # Print the elapsed time
            tqdm.write("@frame {}: distance elapsed time for thread count = {}: {:.4f}s".format(frame,str(self.threads),elapsed_time))

        # relabel the input
        start_time = time.time()

        relabelFilter = sitk.RelabelComponentImageFilter()
        label_sitk_0 = relabelFilter.Execute(sitk.GetImageFromArray(label_0))
        label_0 = sitk.GetArrayFromImage(label_sitk_0)

        label_sitk_1 = relabelFilter.Execute(sitk.GetImageFromArray(label_1))
        label_1 = sitk.GetArrayFromImage(label_sitk_1)

        if self.verbose:
            end_time = time.time()
            # Calculate elapsed time
            elapsed_time = end_time - start_time

            # Print the elapsed time
            tqdm.write("@frame {}: relabel elapsed time for thread count = {}: {:.4f}s".format(frame,str(self.threads),elapsed_time))

        # relabel the input
        start_time = time.time()

        # cell shape measurement by SITK
        statFilter_0 = sitk.LabelShapeStatisticsImageFilter()
        statFilter_0.Execute(label_sitk_0)

        statFilter_1 = sitk.LabelShapeStatisticsImageFilter()
        statFilter_1.Execute(label_sitk_1)

        if self.verbose:
            end_time = time.time()
            # Calculate elapsed time
            elapsed_time = end_time - start_time
            # Print the elapsed time
            tqdm.write("@frame {}: labelShapeStat elapsed time for thread count = {}: {:.4f}s".format(frame,str(self.threads),elapsed_time))

        start_time = time.time()
        overlap_matrix_device = cle.generate_binary_overlap_matrix(label_0, label_1)

        masked_distance_matrix = cle.multiply_images(overlap_matrix_device,distance_matrix_device)

        pointlist = np.concatenate([centroids_0,centroids_1],axis=1)
        distance_matrix_pivot = np.zeros((c_count[0]+c_count[1]+1,c_count[0]+c_count[1]+1))

        distance_matrix_pivot[(centroids_0.shape[1]+1):,1:(centroids_0.shape[1]+1)] = masked_distance_matrix[1:,1:]
        distance_matrix_pivot[1:(centroids_0.shape[1]+1),(centroids_0.shape[1]+1):] = masked_distance_matrix[1:,1:].T

        distance_matrix_pivot = cle.push(distance_matrix_pivot)
        distance_mesh_device = cle.create_labels_like(label_0)
        cle.touch_matrix_to_mesh(pointlist, distance_matrix_pivot,distance_mesh_device)

        if self.verbose:
            end_time = time.time()
            # Calculate elapsed time
            elapsed_time = end_time - start_time
            # Print the elapsed time
            tqdm.write("@frame {}: contact analysis elapsed time for thread count = {}: {:.4f}s".format(frame,str(self.threads),elapsed_time))

        start_time = time.time()
        networkx_graph_two_cell_types_overlap = cle.to_networkx(distance_matrix_pivot, pointlist)
        graph = networkx_graph_two_cell_types_overlap # networkx graph

        if self.verbose:
            end_time = time.time()
            # Calculate elapsed time
            elapsed_time = end_time - start_time
            # Print the elapsed time
            tqdm.write("@frame {}: network export elapsed time for thread count = {}: {:.4f}s".format(frame,str(self.threads),elapsed_time))

        start_time = time.time()
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

        if self.verbose:
            end_time = time.time()
            # Calculate elapsed time
            elapsed_time = end_time - start_time
            # Print the elapsed time
            tqdm.write("@frame {}: device to host elapsed time for thread count = {}: {:.4f}s".format(frame,str(self.threads),elapsed_time))

            end_time = time.time()
            # Calculate elapsed time
            elapsed_time = end_time - start_time_0
            # Print the elapsed time
            tqdm.write("@frame {}: threaded loop elapsed time for thread count = {}: {:.4f}s".format(frame,str(self.threads),elapsed_time))

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

        return {
            "graph":graph, 
            "network_image": distance_mesh_host, 
            "frame": frame, 
            "contact": contact, 
            "contact_label": contact_label, 
            "closest_cell_dist": closest_cell_dist,
            }
    
    def mpi_worker(self):
        # worker function for parallel MPI run.

        from mpi4py import MPI
        # Get our MPI communicator, our rank, and the world size.
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        mpi_size = mpi_comm.Get_size()
        if self.verbose:
            print("{}: Staring mpi worker function at rank {}".format(datetime.now(),mpi_rank))

        req_list = []
        while True:
            if self.verbose:
                print("{}: Calling mpi worker function at rank {}".format(datetime.now(),mpi_comm.Get_rank()))

            # Receive the serialized data
            data_serialized = mpi_comm.recv(source=0)

            # data = mpi_comm.recv(source=0,tag=self.MPI_TAGS["TASK"]) # blocking for immediate consumption of data
            if data_serialized is None:
                # wait all process to finish
                for req in req_list:
                    req.wait()
                
                if self.verbose:
                    print('{}: Rank {} cycle finish'.format(datetime.now(),mpi_comm.Get_rank()))
                return
            
            # Deserialize the data
            data = {}
            if "label_shape" in data_serialized.keys():
                label_shape = np.frombuffer(data_serialized["label_shape"], dtype=np.uint16)

            for key, value in data_serialized.items():
                if key in ["frame"]:
                    # unserialized data
                    data[key] = value
                elif key in ["label_0", "label_1"]:
                    # numpy data need to be reshape
                    data[key] = np.frombuffer(value, dtype=np.uint16).reshape(label_shape)
                elif key in ["label_shape"]:
                    # numpy data no need to reshape
                    data[key] = np.frombuffer(value, dtype=np.uint16)
                elif key in ["features_0", "features_1"]:
                    # convert dict back to pandas df
                    data[key] = pd.DataFrame.from_dict(value)
                else:
                    data[key] = pickle.loads(value)
                    
            if self.verbose:
                print('{}: Rank {} deserialized data @frame {}'.format(datetime.now(),mpi_comm.Get_rank(),data["frame"]))
           
            # unpack the income data to single thread worker
            frame = data["frame"]
            label_0 = data["label_0"]
            label_1 = data["label_1"]
            centroids_0 = data["centroids_0"]
            centroids_1 = data["centroids_1"]
            features_0 = data["features_0"]
            features_1 = data["features_1"]

            # start processing the data
            res = self.run_single_frame(
                label_0=label_0,
                label_1=label_1,
                centroids_0=centroids_0,
                centroids_1=centroids_1,
                features_0=features_0,
                features_1=features_1,
                frame=frame
            )

            if self.verbose:
                print('{}: Rank {} finished data @frame: {}'.format(datetime.now(),mpi_comm.Get_rank(),frame))

            # send finished data back to main process, non blocking for quick receiving for new data, send data will be in the MPI buffer
            # serialize the data

            res_serialized = {}
            for key, value in res.items():
                if key in ["graph"]:
                    res_serialized[key] = pickle.dumps(value)
                elif key in ["frame", "contact", "contact_label", "closest_cell_dist"]:
                    res_serialized[key] = value
                elif key in ["network_image"]:
                    # numpy data
                    res_serialized[key] = value.tobytes()
                else:
                    pass

            if self.verbose:
                print('{}: Rank {} sending data @frame: {}'.format(datetime.now(),mpi_comm.Get_rank(),frame))
            req_list.append(mpi_comm.isend(res_serialized,dest=0,tag=self.MPI_TAGS["RESULT"]))
            if self.verbose:
                print('{}: Rank {} finished sending data @frame: {}'.format(datetime.now(),mpi_comm.Get_rank(),frame))

    def __call__(self, data) -> Any:
        if self.parallel_backend != "MPI":
            labels = data["labels"]
            features = data["features"]
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
                    labels0_ = relabelFilter.Execute(sitk.GetImageFromArray(labels[0][:,:,CUR_FRAME]))
                    labels[0][:,:,CUR_FRAME] = sitk.GetArrayFromImage(labels0_)

                    labels1_ = relabelFilter.Execute(sitk.GetImageFromArray(labels[1][:,:,CUR_FRAME]))
                    labels[1][:,:,CUR_FRAME] = sitk.GetArrayFromImage(labels1_)

                    statFilter = sitk.LabelShapeStatisticsImageFilter()
                    statFilter.Execute(labels0_)

                    statFilter = sitk.LabelShapeStatisticsImageFilter()
                    statFilter.Execute(labels1_)

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
                raise NotImplementedError("PATHOS/PYTHON native parallel backend currently not supported with pyclesperanto")
                # outputs
                network_image = np.zeros_like(labels[0])
                # initiate dicts for unsorted results
                graph_ = {}
                contact_ = {}
                contact_label_ = {}
                closest_cell_dist_ = {}

                def run_single_frame_helper(input_dict):
                    return self.run_single_frame(**input_dict)

                input_dict_list = []

                # TODO: expected to have large memory usage, chuck processing required

                pbar_0 = tqdm(range(START_FRAME,END_FRAME+1),desc="Preparing data for cross cell contact measurements") 
                for CUR_FRAME in pbar_0:
                    input_dict_list.append({
                        "frame": CUR_FRAME,
                        "label_0": labels[0][:,:,CUR_FRAME].compute(),
                        "label_1": labels[1][:,:,CUR_FRAME].compute(),
                        "centroids_0": features[0][features[0].frame==CUR_FRAME][["i","j"]].to_numpy().T,
                        "centroids_1": features[1][features[1].frame==CUR_FRAME][["i","j"]].to_numpy().T,
                        "features_0": features[0],
                        "features_1": features[1],
                    })

                start_time = time.time()
                
                pool = ThreadPool(self.threads)

                results = list(tqdm(pool.uimap(run_single_frame_helper, input_dict_list), total=len(input_dict_list), desc="Cross cell contact measurements (parallel)"))
                # results = list(tqdm(pool.uimap(run_single_frame_helper, input_dict_list, chunksize=32), total=len(input_dict_list), desc="Cross cell contact measurements (parallel)"))
                # results = list(tqdm(pool.amap()))
                
                # close pool and wait to finish
                pool.close()
                pool.join()

                # for res in results:
                #     # network image missing
                #     network_image[:,:,res["frame"]] = res["network_image"]

                #     graph_[res["frame"]] = res["graph"]
                #     contact_[res["frame"]] = res["contact"]
                #     contact_label_[res["frame"]] = res["contact_label"]
                #     closest_cell_dist_[res["frame"]] = res["closest_cell_dist"]

                # reorder unsorted results
                for g in dict(sorted(graph_.items())).values():
                    graph.extend(g)
                for c in dict(sorted(contact_.items())).values():
                    contact.extend(c)
                for c in dict(sorted(contact_label_.items())).values():
                    contact_label.extend(c)
                for c in dict(sorted(closest_cell_dist_.items())).values():
                    closest_cell_dist.extend(c)

            features_out["contact"] = contact
            features_out["contacting cell labels"] = contact_label
            features_out["closest cell dist"] = closest_cell_dist

            return {"image": network_image, "feature": features_out, "network": graph}
        else:
            start_time = time.time()
            from mpi4py import MPI
            # Get our MPI communicator, our rank, and the world size.
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            mpi_size = mpi_comm.Get_size()

            # Do we only have one process?  If yes, then exit.
            if mpi_size == 1:
                print('You are running an MPI program with only one slot/task!')
                print('Are you using `mpirun` (or `srun` when in SLURM)?')
                print('If you are, then please use an `-n` of at least 2!')
                print('(Or, when in SLURM, use an `--ntasks` of at least 2.)')
                print('If you did all that, then your MPI setup may be bad.')
                return 1

            # Is our world size over 999?  The output will be a bit weird.
            # NOTE: Only rank zero does anything, so we don't get duplicate output.
            if mpi_size >= 1000 and mpi_rank == 0:
                print('WARNING:  Your world size {} is over 999!'.format(mpi_size))
                print("The output formatting will be a little weird, but that's it.")

            # Sanity checks complete
            tqdm.write("Parallel run with MPI backend, MPI rank size for GPU processes = {}".format(mpi_size-1))

            # Call the appropriate worker function, based on our rank. Here we use point to point communication with rank 0 as the master program
            if mpi_rank == 0:
                # prepare data for the child process
                images = data["images"]
                labels = data["labels"]
                features = data["features"]
                assert len(images) == 2, "Input images must be 2"
                assert len(features) == 2, "Input features must be 2"
                START_FRAME = features[0].frame.min()
                END_FRAME = features[0].frame.max()
                features_out = features[0].copy()

                # TODO: expected to have large memory usage, chuck processing may be required
                input_dict_list = []

                pbar_0 = tqdm(range(START_FRAME,END_FRAME+1),desc="Preparing data for cross cell contact measurements") 
                for CUR_FRAME in pbar_0:
                    label_0 = labels[0][:,:,CUR_FRAME].compute()
                    label_1 = labels[1][:,:,CUR_FRAME].compute()

                    input_dict_list.append({
                        "frame": CUR_FRAME,
                        "label_shape": np.asarray(label_0.shape).astype(np.uint16),
                        "label_0": label_0,
                        "label_1": label_1,
                        "centroids_0": features[0][features[0].frame==CUR_FRAME][["i","j"]].to_numpy().T,
                        "centroids_1": features[1][features[1].frame==CUR_FRAME][["i","j"]].to_numpy().T,
                        "features_0": features[0],
                        "features_1": features[1],
                    })

                # main program sent dictionary to to child workers
                req_list = []
                for i, x in enumerate(input_dict_list):
                    worker_rank = i%(mpi_size-1)+1
                    if self.verbose:
                        print('{}: Job {} sent data to child rank {}'.format(datetime.now(), i, worker_rank))

                    # Serialize the dictionary
                    data_serialized = {}

                    for key, value in x.items():
                        if key in ["frame"]:
                            # no need for serialization
                            data_serialized[key] = value
                        elif key in ["label_0", "label_1", "label_shape"]:
                            # numpy data
                            data_serialized[key] = value.tobytes()
                        elif key in ["features_0","features_1"]:
                            # pandas table object
                            data_serialized[key] = value.to_dict()
                        else:
                            data_serialized[key] = pickle.dumps(value)

                    # Send the size of the serialized data
                    req_list.append(mpi_comm.isend(data_serialized, dest=worker_rank, tag=self.MPI_TAGS["TASK"]))

                # send finish signal to child workers
                for i in range(mpi_size-1):
                    worker_rank = i%(mpi_size-1)+1
                    if self.verbose:
                        print('{}: Sent end signal to child rank {}'.format(datetime.now(), i, worker_rank))
                    mpi_comm.send(None, dest=worker_rank, tag=self.MPI_TAGS["TASK"]) # send data to child process, blocking to avoid early kill of workers

                # retrieve results from child processes
                graph = []
                network_image = np.zeros_like(labels[0])
                contact = []
                contact_label = []
                closest_cell_dist = []

                # helper dict to collect async results
                graph_ = {}
                contact_ = {}
                contact_label_ = {}
                closest_cell_dist_ = {}

                if self.verbose:
                    print("# of mpi send requests ",len(req_list))

                for req in tqdm(req_list,desc="MPI master data receive progress"):
                    req.wait()
                    status = MPI.Status()
                    res_serialized = mpi_comm.recv(source=MPI.ANY_SOURCE, tag=self.MPI_TAGS["RESULT"], status=status) # gather result here, blocking for immediate consumption, may need optimization
                    source = status.Get_source()

                    if self.verbose:
                        tqdm.write("{}: Master received results from rank {}: {}".format(datetime.now(), source, res_serialized["frame"]))

                    # deserialize the incoming data from workers
                    res = {}
                    for key, value in res_serialized.items():
                        if key in ["graph"]:
                            res[key] = pickle.loads(value)
                        elif key in ["network_image"]:
                            # numpy data need to be reshape
                            res[key] = np.frombuffer(value, dtype=np.uint32).reshape(labels[0].shape[0:2])
                        else:
                            res[key] = value
                    
                    # concat all results
                    # initiate dicts for unsorted results
                    network_image[:,:,res["frame"]] = res["network_image"]
                    graph_[res["frame"]] = res["graph"]
                    contact_[res["frame"]] = res["contact"]
                    contact_label_[res["frame"]] = res["contact_label"]
                    closest_cell_dist_[res["frame"]] = res["closest_cell_dist"]

                # reorder unsorted results
                for g in dict(sorted(graph_.items())).values():
                    graph.extend(g)
                for c in dict(sorted(contact_.items())).values():
                    contact.extend(c)
                for c in dict(sorted(contact_label_.items())).values():
                    contact_label.extend(c)
                for c in dict(sorted(closest_cell_dist_.items())).values():
                    closest_cell_dist.extend(c)

                features_out["contact"] = contact
                features_out["contacting cell labels"] = contact_label
                features_out["closest cell dist"] = closest_cell_dist

                if self.verbose:
                    print("End of MPI master loop")

                if self.verbose:
                    end_time = time.time()
                    # Calculate elapsed time
                    elapsed_time = end_time - start_time
                    # Print the elapsed time
                    tqdm.write("Contact analysis (exclude data concat) elapsed time for mpi gpu worker rank count = {}: {:.4f}s".format(mpi_size-1,elapsed_time))

                return {"image": network_image, "feature": features_out, "network": graph}
            else:
                return