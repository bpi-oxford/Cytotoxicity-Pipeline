import pytest
from utils.label_to_table import *
from aicsimageio import AICSImage

def test_label_to_sparse():
    image_path = "/app/cytotoxicity-pipeline/output/preprocessing/PercentileNormalization/Alive.tif"
    label_path = "/app/cytotoxicity-pipeline/output/segmentation/StarDist/CancerCell.tif"

    image = AICSImage(image_path)
    label = AICSImage(label_path)

    label_np = label.get_image_data("XYT")
    image_np = image.get_image_data("XYT")

    features = label_to_sparse(label=label_np,image=image_np,spacing=[1,1],celltype="")
    # features = label_to_sparse(label=label_np,image=image_np,spacing=[0.8286426,0.8286426],celltype="")

    print(features.head())

    features.to_csv("/app/cytotoxicity-pipeline/output/tracking/Alive.csv",index=False)