import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import numpy as np

def extract_segment_features(image,label,frame,relabel=False,offset=0,cell_type="",spacing=[1,1]):
    image = sitk.GetImageFromArray(image)
    label = sitk.GetImageFromArray(label)
    image.SetSpacing(spacing)
    label.SetSpacing(spacing)

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
            shapeStatFilter.GetPerimeter(i), #perimeter
            imageStatFilter.GetSum(i), #mass
            imageStatFilter.GetMean(i), #mean
            imageStatFilter.GetMedian(i), #median
            imageStatFilter.GetSigma(i), #sd
            frame, #frame
            cell_type, # cell_type
            np.nan, # alive
            ]

    return frame, data, shapeStatFilter.GetLabels()[-1]

def merge_dicts(x,y):
    z = x.copy()
    z.update(y)
    return z

def label_to_sparse(labels, images=None, spacing=[1,1]):
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
        "perimeter",
        "mass",
        "mean",
        "median",
        "sd",
        "frame",
        "cell_type",
        "alive"
        ]

    features = {}

    for k, label in labels.items():
        tqdm.write("Prepare image feature extraction from cell data: {}".format(k))

        pool = Pool(processes=multiprocessing.cpu_count())
        pbar = tqdm(total=label.shape[2])

        results = {}
        def collect_result(result):
            f = pd.DataFrame.from_dict(result[1],orient='index',columns=columns).sort_values(by=["label"])
            results[result[0]] = {"data": f, "offset": result[2]}
            pbar.update(1) # this is just for the fancy progress bar

        for frame in range(label.shape[2]):
            if images:
                image_ = images[k][:,:,frame]
            else:
                image_ = label[:,:,frame]
            label_ = label[:,:,frame]

            # dask array need to compute before apply async
            image_ = image_.compute()
            label_ = label_.compute()
            pool.apply_async(extract_segment_features, args=(image_,label_,frame), kwds={"relabel": True, "offset": 0, "cell_type": k, "spacing": spacing}, callback=collect_result)
        pool.close()
        pool.join()

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
        features[k] = data
    return features