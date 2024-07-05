#@ Float (label="Linking max distance", description="The max distance between two consecutive spots, in physical units, allowed for creating links.", min=1, value=15.0) LINKING_MAX_DISTANCE
#@ int (label="Max frame gap", description="Gap-closing time-distance. The max difference in time-points between two spots to allow for linking. For instance a value of 2 means that the tracker will be able to make a link between a spot in frame t and a successor spots in frame t+2, effectively bridging over one missed detection in one frame.", min=0, value=5) MAX_FRAME_GAP
#@ Float (label="Gap closing distance", description="Gap-closing max spatial distance. The max distance between two spots, in physical units, allowed for creating links over missing detections.", min=1, value=15.0) GAP_CLOSING_MAX_DISTANCE
#@ File (label="Image path", style="file") IMAGE_PATH
#@ File (label="XML path", style="file") XML_PATH
#@ File (label="Output CSV directory", style="directories") OUT_CSV_DIR

import sys
 
from ij import IJ
from ij import WindowManager
 
from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.detection import LabelImageDetectorFactory, ManualDetectorFactory
from fiji.plugin.trackmate.tracking.jaqaman import SimpleSparseLAPTrackerFactory
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettingsIO
from fiji.plugin.trackmate.io import TmXmlReader
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter

from java.io import File

import csv
import os

# IMAGE_PATH = "/home/vpfannenstill/Projects/Cytotoxicity-Pipeline/output/segmentation/StarDist/CancerCell.tif"
# XML_PATH = "/home/vpfannenstill/Projects/Cytotoxicity-Pipeline/output/tracking/trackmate.xml"
# OUT_CSV_DIR = "/home/vpfannenstill/Projects/Cytotoxicity-Pipeline/output/tracking/params"

try:
    if not os.path.exists(OUT_CSV_DIR.getPath()):
        os.makedirs(OUT_CSV_DIR.getPath())
except OSError as e:
    print(e)
OUT_CSV_PATH = os.path.join(OUT_CSV_DIR.getPath(),"trackmate_linkDist-{}_frameGap-{}_gapCloseDist-{}.csv".format(LINKING_MAX_DISTANCE,MAX_FRAME_GAP,GAP_CLOSING_MAX_DISTANCE))
 
# We have to do the following to avoid errors with UTF8 chars generated in 
# TrackMate that will mess with our Fiji Jython.
reload(sys)
sys.setdefaultencoding('utf-8')
 
# Get currently selected image
# imp = WindowManager.getCurrentImage()
# imp = IJ.openImage(IMAGE_PATH)
# imp.show()

#----------------------------
# Load Trackmate XML file
#----------------------------
reader = TmXmlReader(XML_PATH)
if not reader.isReadingOk():
    sys.exit(reader.getErrorMessage())

imp = reader.readImage()
 
#----------------------------
# Create the model object now
#----------------------------
 
# Some of the parameters we configure below need to have
# a reference to the model at creation. So we create an
# empty model now.
 
# model = Model()
model = reader.getModel()
 
# Send all messages to ImageJ log window.
model.setLogger(Logger.IJ_LOGGER)
logger = model.getLogger()

spots = model.getSpots()
logger.log(str(spots))
 
#------------------------
# Prepare settings object
#------------------------
 
settings = Settings(imp)
 
# Configure detector - We use the Strings for the keys
# settings.detectorFactory = LabelImageDetectorFactory()
# settings.detectorSettings = {
#     'TARGET_CHANNEL' : 1,
#     'SIMPLIFY_CONTOURS' : False,
# }  
settings.detectorFactory = ManualDetectorFactory()


# Configure spot filters - Classical filter on quality
#filter1 = FeatureFilter('QUALITY', 30, True)
#settings.addSpotFilter(filter1)
 
# Configure tracker - We want to allow merges and fusions
settings.trackerFactory = SimpleSparseLAPTrackerFactory()
settings.trackerSettings = settings.trackerFactory.getDefaultSettings() # almost good enough
settings.trackerSettings['LINKING_MAX_DISTANCE'] = LINKING_MAX_DISTANCE
settings.trackerSettings['MAX_FRAME_GAP'] = MAX_FRAME_GAP
settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = GAP_CLOSING_MAX_DISTANCE

# settings.trackerFactory = OverlapTrackerFactory()
 
# Add ALL the feature analyzers known to TrackMate. They will 
# yield numerical features for the results, such as speed, mean intensity etc.
# settings.addAllAnalyzers()
 
# Configure track filters - We want to get rid of the two immobile spots at
# the bottom right of the image. Track displacement must be above 10 pixels.
 
#filter2 = FeatureFilter('TRACK_DISPLACEMENT', 10, True)
#settings.addTrackFilter(filter2)
 
 
#-------------------
# Instantiate plugin
#-------------------
 
trackmate = TrackMate(model, settings)
 
#--------
# Process
#--------
 
ok = trackmate.checkInput()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))
 
ok = trackmate.process()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))
 
 
#----------------
# Display results
#----------------
 
# A selection.
selectionModel = SelectionModel( model )
 
# Read the default display settings.
ds = DisplaySettingsIO.readUserDefault()
 
# displayer =  HyperStackDisplayer( model, selectionModel, imp, ds )
# displayer.render()
# displayer.refresh()
 
# Echo results with the logger we set at start:
model.getLogger().log( str( model ) )

# The feature model, that stores edge and track features.
fm = model.getFeatureModel()
 
# Iterate over all the tracks that are visible.
fields = ["ID","POSITION_X","POSITION_Y","FRAME","TRACK_ID"]
rows = []


for id in model.getTrackModel().trackIDs(True):
    # Fetch the track feature from the feature model.
    # v = fm.getTrackFeature(id, 'TRACK_MEAN_SPEED')
    # model.getLogger().log('')
    # model.getLogger().log('Track ' + str(id) + ': mean velocity = ' + str(v) + ' ' + model.getSpaceUnits() + '/' + model.getTimeUnits())
 
	# Get all the spots of the current track.
    track = model.getTrackModel().trackSpots(id)
    for spot in track:
        sid = spot.ID()
        # Fetch spot features directly from spot.
        # Note that for spots the feature values are not stored in the FeatureModel
        # object, but in the Spot object directly. This is an exception; for tracks
        # and edges, you have to query the feature model.
        x=spot.getFeature('POSITION_X')
        y=spot.getFeature('POSITION_Y')
        t=int(spot.getFeature('FRAME'))
        # q=spot.getFeature('QUALITY')
        # snr=spot.getFeature('SNR_CH1')
        # mean=spot.getFeature('MEAN_INTENSITY_CH1')
        # model.getLogger().log('\tspot ID = ' + str(sid) + ': x='+str(x)+', y='+str(y)+', t='+str(t)+', q='+str(q) + ', snr='+str(snr) + ', mean = ' + str(mean))
        rows.append([sid,x,y,t,id])

with open(OUT_CSV_PATH,"wb") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(rows)