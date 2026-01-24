from setuptools import setup, find_packages

setup(
    name = "mcDETECT",
    version = "2.0.14",
    packages = find_packages(),
    install_requires = ["anndata", "miniball", "numpy", "pandas", "rtree", "scanpy", "scikit-learn", "scipy", "shapely"],
    author = "Chenyang Yuan",
    author_email = "chenyang.yuan@emory.edu",
    description = "Uncovering the dark transcriptome in polarized neuronal compartments with mcDETECT",
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    url = "https://github.com/chen-yang-yuan/mcDETECT",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = ">=3.6",
)