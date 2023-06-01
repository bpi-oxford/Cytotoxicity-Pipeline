# Developer Guide for Cytotoxicity Pipeline

Cytotoxicity pipeline is modular based CLI workflow. The workflow is designed as:
1. Preprocessing (image -> image)
2. Segmentation (image -> label / label -> label)
3. Detection (image -> dataframe/ label -> dataframe)
3. Tracking (label -> dataframe/ dataframe -> dataframe)
4. Analysis (dataframe -> dataframe/ arbitrary outputs)

For each workflow process it only accept specific type of input-output pair. To enhance module reusability the data is interfaced as a python dictionary object, e.g. in preprocessing we provide a dict input with keyword "image". Then return file is in the same format.

For individual steps check the table for input-output dictionary key-pairs:

| Step          | Input Key           | Output Key          |
|---------------|---------------------|---------------------|
| Preprocessing | "image"             | "image"             |
| Segmentation  | "image"/"label"     | "label"             |
| Detection     | "image"/"label"     | "dataframe"         |
| Tracking      | "dataframe"         | "dataframe"         |
| Analysis      | "dataframe"         | arbitrary           |

For short, following the template class for the modular workflow:

```python
class MySegmentationClass(object):
    def __init__(self,arg1=[0,1,2],arg2="foo", verbose=True) -> None:
        """
        Function documentation comes to here

        Args:
            arg1 (tuple or int): Tuple or integer argument input
            arg2 (str): String input
            verbose (bool): Turn on or off the processing printout
        """
        self.name = "MyPreprocessingClass"
        self.arg1 = arg1
        self.arg2 = arg2
        self.verbose = verbose

    def __call__(self, data) -> Any:
        image = data["image"]

        if self.verbose:
            tqdm.write("Class args: [{},{}]".format(self.arg1,self.arg2))

        # some processing here
        ...
        label = awesome_segmentation(image)

        return {"image": image, "label": label}

```

A full example of the class can be found in [../preprocessing/normalization.py](../preprocessing/normalization.py)

⚠️ **Important**: Add your package dependency to [requirements.txt](../requirements.txt) ⚠️

⚠️ Add notes to [README.md](../README.md) when necessary, particularly conda/mamba specific dependencies⚠️

To load the class back to the main function you only need to add corresponding header import and edit the pipeline YAML file.

## Custom Intermediate Output
TODO

## Dask Support
For better big data management we recommended the usage of [Dask array](https://docs.dask.org/en/stable/array.html) than numpy, though in some cases cupy and numpy may do the work.