from typing import Any
from tqdm import tqdm
import tempfile
import os
from aicsimageio.writers import OmeTiffWriter

class TrackMate(object):
    def __init__(self, FIJI_DIR="", verbose=True) -> None:
        """
        Perform TrackMate tracking with pyimagej wrapping

        Args:
            FIJI_APP (str): Path to Fiji binary
            verbose (bool): Turn on or off the processing printout
        """
        self.name = "TrackMate"
        self.FIJI_DIR = FIJI_DIR
        self.verbose = verbose

    def __call__(self, data, output=False) -> Any:
        image = data["image"]
        features = data["feature"]

        if self.verbose:
            tqdm.write("Cell tracking with TrackMate")

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
            "--targetFilePath={}".format(temp_xml_file.name),
            # "--targetFilePath={}".format("/app/cytotoxicity-pipeline/output/tracking/trackmate.xml")
            # "--targetFilePath={}".format("/home/jackyko/Projects/Cytotoxicity-Pipeline/output/tracking/trackmate.xml")
        ])

        os.system(conversion_command)

        print("CsvToTrackMate script complete")

        """
        Tracking with TrackMate
        Export TrackMate file as csv
        """

        # print("Image loading on python side...")
        # image = io.imread(IMAGE_PATH)
        # imp = ij.py.to_imageplus(image)

        # File = sj.jimport('java.io.File')
        # jfile = File(os.path.join(CWD,'trackmate.py'))

        # # convert python side stuffs to java side
        # # define python side arguments
        # args = {
        #     'imp': imp, # java side ImagePlus object
        #     'headless': headless
        # }
        # jargs = ij.py.jargs(args)
        # result_future = ij.script().run(jfile,True,jargs)

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
