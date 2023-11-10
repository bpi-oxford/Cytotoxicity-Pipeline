import pytest
from tracking.trackmate import *
import pandas as pd
from aicsimageio import AICSImage

def test_trackmate():
    image_path = "/app/cytotoxicity-pipeline/output/preprocessing/PercentileNormalization/Alive.tif"
    feature_path = "/app/cytotoxicity-pipeline/output/tracking/Alive.csv"

    img = AICSImage(image_path)  # selects the first scene found

    # Pull only a specific chunk in-memory
    img_dask = img.get_image_dask_data("XYT")

    data = {
        "image": img_dask.persist(),
        "feature": pd.read_csv(feature_path)
    }

    tm = TrackMate(FIJI_DIR="/app/Fiji.app")
    tm(data)