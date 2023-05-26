import os
import sys
import argparse
import yaml
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from dask_init import *
from preprocessing.normalization import *

def get_args():
	parser = argparse.ArgumentParser(description="Inference script for 3D cell classifier")

	parser.add_argument(
		'-v', '--verbose',
		dest='verbose',
		help='Show verbose output',
		action='store_true')
	parser.add_argument(
		'-p','--pipeline',
		dest='pipeline',
		help='Path to the pipeline YAML',
		metavar='PATH',
		required=True)

	args = parser.parse_args()

	# print arguments if verbose
	if args.verbose:
		args_dict = vars(args)
		for key in sorted(args_dict):
			print("{} = {}".format(str(key), str(args_dict[key])))

	return args

def main(args):
	pipeline_file = args.pipeline

	with open(pipeline_file, 'r') as f:
		pipeline = yaml.safe_load(f)

	# initiate dask cluster
	client = init_dask_cluster()
	
	# data loading
	images = {}
	for ch, path in pipeline["channels"].items():
		# Get an AICSImage object
		img = AICSImage(path)  # selects the first scene found

		# Pull only a specific chunk in-memory
		slice_x = slice(pipeline["image_range"]["x"][0],pipeline["image_range"]["x"][1],pipeline["image_range"]["x"][2])
		slice_y = slice(pipeline["image_range"]["y"][0],pipeline["image_range"]["y"][1],pipeline["image_range"]["y"][2])
		slice_t = slice(pipeline["image_range"]["t"][0],pipeline["image_range"]["t"][1],pipeline["image_range"]["t"][2])
		img_dask = img.get_image_dask_data("XYT")[slice_x,slice_y,slice_t]
		images[ch] = img_dask.persist()

	# create output dir
	os.makedirs(pipeline["output_dir"],exist_ok=True)

	# preprocessing
	for p in pipeline['pipeline']["preprocessing"]:
		class_name = p['name']
		class_args = p["args"]
				
		# # Dynamically instantiate the class
		class_obj = globals()[p["name"]](**class_args)
		if p["channels"] == "all":
			channels = pipeline["channels"].keys()
		else:
			channels = p["channels"]

		for ch in channels:
			# inplace update
			res = class_obj({"image": images[ch]})
			images[ch] = res["image"]

			# export to ome tiff format
			if p["output"]:
				output_dir = os.path.join(pipeline["output_dir"],"preprocessing",class_name)
				output_file = os.path.join(output_dir,"{}.tif".format(ch))
				os.makedirs(output_dir,exist_ok=True)
				tqdm.write("Exporting result: {}".format(output_file))
				OmeTiffWriter.save(images[ch], output_file, dim_order="TYX")

	# segmentation

if __name__ == "__main__":
	args = get_args()
	main(args)
