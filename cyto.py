import os
import sys
import argparse
import yaml
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from init import *
from preprocessing.normalization import *
from segmentation.stardist import *
from tracking.trackmate import *
from utils.label_to_table import *
from utils.utils import *

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

def preprocessing(images, pipeline):
	for p in pipeline['pipeline']["preprocessing"]:
		class_name = p['name']
		class_args = p["args"]
				
		# Dynamically instantiate the class
		class_obj = globals()[p["name"]](**class_args)
		if p["channels"] == "all":
			channels = images.keys()
		else:
			channels = p["channels"]

		for ch in channels:
			if isinstance(ch, str):
				# inplace update
				tqdm.write("Channel: {}".format(ch))
				res = class_obj({"image": images[ch]})
				images[ch] = res["image"]
			elif isinstance(ch, list):
				# multiple input
				tqdm.write("Multi channel input: [{}]".format(', '.join([str(ch_) for ch_ in ch])))
				images_ = [images[ch_] for ch_ in ch]
				res = class_obj({"images": images_}) # beware of the difference btw "image" and "images" keys
				images[p["output_channel_name"]] = res["output"]

			# export to ome tiff format
			if p["output"]:
				output_dir = os.path.join(pipeline["output_dir"],"preprocessing",class_name)
				if isinstance(ch, str):
					output_file = os.path.join(output_dir,"{}.tif".format(ch))
					os.makedirs(output_dir,exist_ok=True)
					tqdm.write("Exporting result: {}".format(output_file))
					OmeTiffWriter.save(images[ch].T, output_file, dim_order="TYX")
				elif isinstance(ch, list):
					output_file = os.path.join(output_dir,"{}.tif".format(p["output_channel_name"]))
					os.makedirs(output_dir,exist_ok=True)
					tqdm.write("Exporting result: {}".format(output_file))
					OmeTiffWriter.save(images[p["output_channel_name"]].T, output_file, dim_order="TYX")

	return images

def segmentation(images, pipeline):
	labels = {}
	for p in pipeline["pipeline"]["segmentation"]:
		class_name = p["name"]
		class_args = p["args"]

		input_type = p["input_type"] # image, label
		output_type = p["output_type"] # label

		# Dynamically instantiate the class
		if class_args:
			class_obj = globals()[p["name"]](**class_args)
		else:
			class_obj = globals()[p["name"]]()
		if p["channels"] == "all":
			channels = images.keys()
		else:
			channels = p["channels"]

		for ch in channels:
			# inplace update
			tqdm.write("Channel: {}".format(ch))
			if input_type == "image":
				input = images
			elif input_type == "label":
				input = labels
			res = class_obj({input_type: input[ch]})
			labels[ch] = res["label"]

			# export to ome tiff format
			if p["output"]:
				output_dir = os.path.join(pipeline["output_dir"],"segmentation",class_name)
				output_file = os.path.join(output_dir,"{}.tif".format(ch))
				os.makedirs(output_dir,exist_ok=True)
				tqdm.write("Exporting result: {}".format(output_file))
				OmeTiffWriter.save(labels[ch].T, output_file, dim_order="TYX")

	return images, labels

def main(args):
	pipeline_file = args.pipeline

	with open(pipeline_file, 'r') as f:
		pipeline = yaml.safe_load(f)

	# initiate dask cluster
	client = init_dask_cluster()

	# initiate fiji
	pyimagej_init(FIJI_DIR=pipeline["fiji_dir"])
	
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
	if pipeline["pipeline"]["preprocessing"]:
		images = preprocessing(images,pipeline)

	# segmentation
	if pipeline["pipeline"]["segmentation"]:
		images, labels = segmentation(images,pipeline)

	# convert segmentation mask to trackpy style array
	features = {}

	print("images: ", images.keys())
	print("labels: ", labels.keys())

	for image_ch, label_ch in pipeline["pipeline"]["label_to_sparse"]["image_label_pair"]:
		tqdm.write("Converting segmentation mask to sparse table...")
		tqdm.write("Image Channel: {}".format(image_ch))
		tqdm.write("Label Channel: {}".format(label_ch))

		# TODO: fix spacing issue
		# features[image_ch] = label_to_sparse(label=labels[label_ch],image=images[image_ch],spacing=pipeline["spacing"],celltype=image_ch)
		features[image_ch] = label_to_sparse(label=labels[label_ch],image=images[image_ch],spacing=[1,1],celltype=image_ch)

		output_dir = os.path.join(pipeline["output_dir"],"tracking")
		output_file = os.path.join(output_dir,"{}.csv".format(image_ch))
		os.makedirs(output_dir,exist_ok=True)
		tqdm.write("Exporting result: {}".format(output_file))
		features[image_ch].to_csv(output_file,index=False)

	# tracking
	p = pipeline["pipeline"]["tracking"][0]
	class_name = p["name"]
	class_args = p["args"]
	class_args["FIJI_DIR"] = pipeline["fiji_dir"]

	# Dynamically instantiate the class
	if class_args:
		class_obj = globals()[p["name"]](**class_args)
	else:
		class_obj = globals()[p["name"]]()
	if p["channels"] == "all":
		channels = images.keys()
	else:
		channels = p["channels"]

	for ch in channels:
		tqdm.write("Tracking channel: {}".format(ch))
		res = class_obj({"image": images[ch], "feature": features[ch]})

if __name__ == "__main__":
	args = get_args()
	main(args)
