from typing import Any
from tqdm import tqdm
import tempfile

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

    def __call__(self, data) -> Any:
        image = data["image"]
        features = data["feature"]

        print(features.columns)

        if self.verbose:
            tqdm.write("Cell tracking with TrackMate")

        # convert csv to trackmate xml
        # Create a temporary saved csv file
        temp_csv_file = tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False)
        features.to_csv(temp_csv_file.name, index=False)

        conversion_command = " ".join([
            os.path.join(self.FIJI_DIR,),
            "--headless",
            os.path.join(self.FIJI_DIR,"scripts","CsvToTrackMate.py"), 
            "--csvFilePath={}".format(temp_csv_file) 
            # --imageFilePath="/path/to/MyImage.tif"
            # --xCol=1 
            # --radius=2 
            # --yCol=2 
            # --zCol=3 
            # --frameCol=0
            # --targetFilePath="/path/to/TrackMateFile.xml"
        ])


        # Close the temporary files
        temp_file.close()

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

        return None
