from typing import Any
from tqdm import tqdm
import tempfile
import os
from aicsimageio.writers import OmeTiffWriter
import scyjava as sj
from scyjava import jimport
import imagej
import imagej.doctor

class TrackMate(object):
    def __init__(self, FIJI_DIR="", ij=None, linking_max_distance=15.0, max_frame_gap=5, gap_closing_max_distance=15.0, size_min=None, verbose=True) -> None:
        """
        Perform TrackMate tracking with pyimagej wrapping

        Args:
            FIJI_APP (str): Path to Fiji.app folder
            ij (imagej object): pyimagej object, if not provided a new one will be initiated from the given FIJI_APP path
            linking_max_distance (float): The max distance between two consecutive spots, in physical units, allowed for creating links.
            max_frame_gap (int): Gap-closing time-distance. \
                The max difference in time-points between two spots to allow for linking. \
                For instance a value of 2 means that the tracker will be able to make a link \
                between a spot in frame t and a successor spots in frame t+2, effectively \
                bridging over one missed detection in one frame.
            gap_closing_max_distance (float): Gap-closing max spatial distance. The max distance between two spots, in physical units, allowed for creating links over missing detections.
            size_min (int): Minimum cell size to filter out small cells
            verbose (bool): Turn on or off the processing printout
        """
        self.name = "TrackMate"
        self.FIJI_DIR = FIJI_DIR
        self.ij = ij
        self.linking_max_distance = linking_max_distance
        self.max_frame_gap = max_frame_gap
        self.gap_closing_max_distance = gap_closing_max_distance
        self.size_min = size_min
        self.verbose = verbose

    def __call__(self, data, output=False) -> Any:
        image = data["image"]
        features = data["feature"]

        if self.verbose:
            tqdm.write("Cell tracking with TrackMate")

        # filter by cell features
        if self.size_min is not None:
            features = features[features["size"]>self.size_min]

        # convert csv to trackmate xml
        # Create a temporary saved csv file
        temp_csv_file = tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False)
        temp_img_file = tempfile.NamedTemporaryFile(suffix=".tiff", mode="w", delete=False)
        temp_xml_file = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
        features.to_csv(temp_csv_file.name, index=False)

        OmeTiffWriter.save(image.T, temp_img_file.name, dim_order="TYX")

        conversion_command = " ".join([
            os.path.join(self.FIJI_DIR,"ImageJ-linux64"),
            "--headless",
            os.path.join(self.FIJI_DIR,"scripts","CsvToTrackMate.py"), 
            "--csvFilePath={}".format(temp_csv_file.name),
            "--imageFilePath={}".format(temp_img_file.name),
            "--radius=2.5",
            "--xCol={}".format(list(features.columns).index("j")),
            "--yCol={}".format(list(features.columns).index("i")),  
            "--frameCol={}".format(list(features.columns).index("frame")),
            "--idCol={}".format(list(features.columns).index("label")),
            "--nameCol={}".format(list(features.columns).index("cell_type")),
            "--radiusCol={}".format(list(features.columns).index("feret_radius")),
            # "--targetFilePath={}".format(temp_xml_file.name),
            # "--targetFilePath={}".format("/app/cytotoxicity-pipeline/output/tracking/trackmate.xml")
            "--targetFilePath={}".format("/home/vpfannenstill/Projects/Cytotoxicity-Pipeline/output/tracking/trackmate.xml")
        ])

        os.system(conversion_command)

        print("CsvToTrackMate script complete")

        """
        Tracking with TrackMate
        Export TrackMate file as csv
        """
        if self.ij is None:
            # initiate Fiji
            imagej.doctor.checkup()

            if not os.path.exists(self.FIJI_DIR) or self.FIJI_DIR == "":
                raise Exception("Fiji.app directory not found")

            print("Initializing Fiji on JVM...")
            self.ij = imagej.init(self.FIJI_DIR,mode='headless')
            print(self.ij.getApp().getInfo(True))

        # print("Image loading on python side...")
        # # image = io.imread(IMAGE_PATH)
        # imp = ij.py.to_imageplus(image)

        File = sj.jimport('java.io.File')
        CWD = os.path.dirname(os.path.realpath(__file__))
        jfile = File(os.path.join(CWD,'trackmate_script_run.py'))

        # convert python side stuffs to java side
        # define python side arguments
        args = {
            'LINKING_MAX_DISTANCE': self.linking_max_distance, 
            'MAX_FRAME_GAP': self.max_frame_gap,
            'GAP_CLOSING_MAX_DISTANCE': self.gap_closing_max_distance
        }

        jargs = self.ij.py.jargs(args)
        print("Running Trackmate...")
        result_future = self.ij.script().run(jfile,True,jargs)

        # # get the result from java future, blocking will occur here
        # result = result_future.get()
        # if not headless:
        #     input("Press Enter to continue...")
        # print(ij.py.from_java(result.getOutput("foo")))
        # print(ij.py.from_java(result.getOutput("bar")))
        # print(ij.py.from_java(result.getOutput("shape")))

        if output:
            # Reading the data inside the xml
            # file to a variable under the name
            # data
            with open(temp_xml_file.name, 'r') as f:
                data = f.read()

            # Close the temporary files
            temp_csv_file.close()
            temp_img_file.close()
            temp_xml_file.close()

            return features, data
        else:
            # Close the temporary files
            temp_csv_file.close()
            temp_img_file.close()
            temp_xml_file.close()

            return features, None

def main():
    from aicsimageio import AICSImage
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import ParameterGrid

    FIJI_DIR = "/home/vpfannenstill/Fiji.app"

    image_ = AICSImage("/home/vpfannenstill/Projects/Cytotoxicity-Pipeline/output/preprocessing/PercentileNormalization/CancerCell.tif")
    image = image_.get_image_data("XYT")
    feature = pd.read_csv("/home/vpfannenstill/Projects/Cytotoxicity-Pipeline/output/tracking/Alive.csv")
    feature["cell_type"] = np.nan
    
    param_grid = {
        "linking_max_distance": range(20,26,2),
        "max_frame_gap": range(4,6,1),
        "gap_closing_max_distance": range(10,26,2),
        }
    
    params = list(ParameterGrid(param_grid))

    # initiate Fiji
    imagej.doctor.checkup()

    if not os.path.exists(FIJI_DIR) or FIJI_DIR == "":
        raise Exception("Fiji.app directory not found")

    print("Initializing Fiji on JVM...")
    ij = imagej.init(FIJI_DIR,mode='headless')
    print(ij.getApp().getInfo(True))

    pbar = tqdm(params[0:])
    for param in pbar:
        tm = TrackMate(FIJI_DIR=FIJI_DIR,
                       ij=ij,
                       gap_closing_max_distance=param["gap_closing_max_distance"],
                       linking_max_distance=param["linking_max_distance"],
                       max_frame_gap=param["max_frame_gap"],
                       size_min=3
                       )
        tm({"image": image, "feature":feature})

if __name__ == "__main__":
    main()