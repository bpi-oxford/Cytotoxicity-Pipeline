from setuptools import setup, find_packages

# loading packaged dependency from requirements.txt
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="pyCyto",
    version="0.1.0",
    packages=find_packages(include=[
        "cyto",
        "cyto.*"
    ]),
    install_requires=required,
    entry_points={
        "console_scripts":[
            "cyto = main:main",
        ]
    },
    package_data={
        "": ["desc.txt"]
    }
)