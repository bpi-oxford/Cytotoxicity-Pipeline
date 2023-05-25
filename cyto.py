import os
import sys
import argparse
import yaml

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
                
	print(pipeline['pipeline'])
	
	# preprocessing
	for p in pipeline['pipeline']["preprocessing"]:
		class_name = p['name']
		class_args = p["args"]

		print(class_name)
		print(class_args)
                
		# # Dynamically instantiate the class
    	# class_obj = globals()[class_name](**class_args)

if __name__ == "__main__":
    args = get_args()
    main(args)
