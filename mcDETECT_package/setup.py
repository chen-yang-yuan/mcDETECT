from setuptools import setup, find_packages

setup(
    name = "mcDETECT",
    version = "1.0.9",
    packages = find_packages(),
    install_requires = ["anndata", "miniball", "numpy", "pandas", "rtree", "scanpy", "scikit-learn", "scipy", "shapely"],
    author = "Chenyang Yuan",
    author_email = "chenyang.yuan@emory.edu",
    description = "mcDETECT: Decoding 3D Spatial Synaptic Transcriptomes with Subcellular-Resolution Spatial Transcriptomics",
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