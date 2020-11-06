from setuptools import find_packages, setup

with open("AUTHORS.txt", "r") as f:
    authors = f.readline()

setup(
    name="genni",
    version="1.0.0",
    author=authors,
    author_email="isak.falk@live.se",
    description="GENNI: Visualising the Geometry of Equivalences for Neural Network Identifiability",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pyyaml",
        "torch",
        "torchvision",
        "ray[tune]",
        "numpy",
        "matplotlib",
        "pandas",
    ],
)
