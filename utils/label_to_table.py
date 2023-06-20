import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import numpy as np
import dask.array as da
import threading
from threading import Lock

mutex = Lock()

def extract_segment_features(image,label,frame,relabel=False,offset=0,cell_type="",spacing=[1,1]):
    mutex.acquire()
    try: 
        image = sitk.GetImageFromArray(image)
        label = sitk.GetImageFromArray(label)
        label = sitk.Cast(label,sitk.sitkUInt16)
        image.SetSpacing(tuple(spacing))
        label.SetSpacing(tuple(spacing))

        if relabel:
            relabelFilter = sitk.RelabelComponentImageFilter()
            label = relabelFilter.Execute(label)

        shapeStatFilter = sitk.LabelShapeStatisticsImageFilter()
        shapeStatFilter.ComputeFeretDiameterOn()
        shapeStatFilter.Execute(label)

        imageStatFilter = sitk.LabelStatisticsImageFilter()
        imageStatFilter.Execute(image,label)

        data={}

        for i in shapeStatFilter.GetLabels():
            bbox_start = label.TransformPhysicalPointToContinuousIndex((shapeStatFilter.GetBoundingBox(i)[0],shapeStatFilter.GetBoundingBox(i)[1]))
            bbox_end = label.TransformPhysicalPointToContinuousIndex((
                    shapeStatFilter.GetBoundingBox(i)[0] + shapeStatFilter.GetBoundingBox(i)[2],
                    shapeStatFilter.GetBoundingBox(i)[1] + shapeStatFilter.GetBoundingBox(i)[3],
                ))

            data[str(i+offset)] = [
                i+offset, # label
                shapeStatFilter.GetCentroid(i)[1], #y
                shapeStatFilter.GetCentroid(i)[0], #x
                shapeStatFilter.GetBoundingBox(i)[0], # bbox_xstart
                shapeStatFilter.GetBoundingBox(i)[1], # bbox_ystart
                shapeStatFilter.GetBoundingBox(i)[2], # bbox_xsize
                shapeStatFilter.GetBoundingBox(i)[3], # bbox_ysize
                label.TransformPhysicalPointToContinuousIndex(shapeStatFilter.GetCentroid(i))[0], # i
                label.TransformPhysicalPointToContinuousIndex(shapeStatFilter.GetCentroid(i))[1], # j
                bbox_start[0], # bbox_istart
                bbox_start[1], # bbox_jstart
                bbox_end[0] - bbox_start[0], # bbox_isize
                bbox_end[1] - bbox_start[1], # bbox_jsize
                shapeStatFilter.GetPhysicalSize(i), # size
                shapeStatFilter.GetElongation(i), #elongation
                shapeStatFilter.GetFlatness(i), #flatness
                shapeStatFilter.GetRoundness(i), #roundness
                shapeStatFilter.GetFeretDiameter(i), #feret diameter
                shapeStatFilter.GetFeretDiameter(i)/2, #feret radius
                shapeStatFilter.GetPerimeter(i), #perimeter
                imageStatFilter.GetSum(i), #mass
                imageStatFilter.GetMean(i), #mean
                imageStatFilter.GetMedian(i), #median
                imageStatFilter.GetSigma(i), #sd
                frame, #frame
                cell_type, # cell_type
                np.nan, # alive
                ]

        mutex.release()
        print(frame, shapeStatFilter.GetLabels()[-1])
        
        return frame, data, shapeStatFilter.GetLabels()[-1]
    except Exception as e:
        mutex.release()
        print(e)
        return frame, {}, 0
    
def merge_dicts(x,y):
    z = x.copy()
    z.update(y)
    return z

def label_to_sparse(label, image=None, spacing=[1,1],celltype=""):
    # extracting the segment centroids
    columns = [
        "label",
        "y",
        "x",
        "bbox_xstart",
        "bbox_ystart",
        "bbox_xsize",
        "bbox_ysize",
        "i",
        "j",
        "bbox_istart",
        "bbox_jstart",
        "bbox_isize",
        "bbox_jsize",
        "size",
        "elongation",
        "flatness",
        "roundness",
        "feret_diameter",
        "feret_radius",
        "perimeter",
        "mass",
        "mean",
        "median",
        "sd",
        "frame",
        "cell_type",
        "alive"
        ]

    pbar = tqdm(total=label.shape[2])

    results = {}
    def collect_result(result):
        # print("collecting data: {}".format(len(result[1])))
        try:
            f = pd.DataFrame.from_dict(result[1],orient='index',columns=columns).sort_values(by=["label"])
            # print("complete pd to df")
            results[result[0]] = {"data": f, "offset": result[2]}
        except Exception as e:
            print(e)

        pbar.update(1) # this is just for the fancy progress bar

    # pool = Pool(processes=multiprocessing.cpu_count())
    pool = Pool(processes=1)
    process_data = []
    results_iter = []
    for frame in range(label.shape[2]):
        if image is not None:
            image_ = image[:,:,frame]
        else:
            image_ = label[:,:,frame]
        label_ = label[:,:,frame]

        # dask array need to compute before apply async
        if isinstance(image_, da.Array):
            image_ = image_.compute()
        if isinstance(label_,da.Array):
            label_ = label_.compute()
        
        # for unknown reason multi threaded process get dead lock for certain process, unable to fix
        # pool.apply_async(extract_segment_features, args=(image_,label_,frame), kwds={"relabel": True, "offset": 0, "cell_type": celltype, "spacing": spacing}, callback=collect_result)
        results_iter.append(extract_segment_features(image_,label_,frame, relabel=True, offset=0, cell_type=celltype, spacing=spacing))
    pool.close()
    pool.join()

    [collect_result(result) for result in results_iter]
        
    # collect data to dictionary form
    # tqdm.write("Applying label number offset")
    data = pd.DataFrame(columns=columns)
    offset = 0
    data = []
    for key in tqdm(sorted(results)):
        data_ = results[key]["data"]
        data_["label"] = data_['label'].apply(lambda x: x + offset)
        # data = merge_dicts(data,data_)
        data.append(data_)
        offset += results[key]["offset"]

    data = pd.concat(data,ignore_index=True)
    return data