# mcDETECT v2.0.15

## Uncovering the dark transcriptome in polarized neuronal compartments with mcDETECT

#### Chenyang Yuan, Krupa Patel, Hongshun Shi, Hsiao-Lin V. Wang, Feng Wang, Ronghua Li, Yangping Li, Victor G. Corces, Hailing Shi, Sulagna Das, Jindan Yu, Peng Jin, Bing Yao* and Jian Hu*

mcDETECT is a computational framework designed to study the dark transcriptome related to polarized compartments in brain using *in situ* spatial transcriptomics (iST) data. It begins by examining the subcellular distribution of mRNAs in an iST sample. Each mRNA molecule is treated as a distinct point with its own 3D spatial coordinates considering the thickness of the sample. Unlike many cell-type marker genes, which are typically found within the nucleus or soma, compartmentalized mRNAs often form small aggregates outside the soma. mcDETECT uses a density-based clustering approach to identify these extrasomatic aggregates. This involves calculating the Euclidean distance between mRNA points and defining the neighborhood of each point within a specified search radius. Points are then categorized as core points, border points, or noise points based on their reachability from neighboring points. mcDETECT recognizes each connected bundle of core and border points as a mRNA aggregate. To minimize false positives, it excludes aggregates that substantially overlap with somata, which are estimated by dilating the nuclear masks derived from DAPI staining. mcDETECT then repeats this process for multiple granule markers, merging aggregates from different markers that exhibit high spatial overlap. After aggregating across all markers, an additional filtering step removes aggregates containing mRNAs from negative control genes, which are known to be enriched exclusively in nuclei and somata. The remaining aggregates are considered individual RNA granules. mcDETECT then computes the minimum enclosing sphere for each aggregate to connect neighboring mRNA molecules from all measured genes and summarizes their counts, thereby defining the spatial transcriptome profile of individual RNA granules.

![mcDETECT workflow](docs/workflow.jpg)<br>

## Installation

To install `mcDETECT` package you must make sure that your python version is over 3.6. If you don’t know the version of python you can check it by:

```python
import platform
platform.python_version()
```

Now you can install the current release of `mcDETECT` by the following three ways:

### 1. PyPI

Directly install the package from PyPI:

```bash
# Note: you need to make sure that the pip is for python3
pip3 install mcDETECT

# If you do not have permission (when you get a permission denied error), you should run:
pip3 install --user mcDETECT

# Or you could run:
python3 -m pip install mcDETECT
```

### 2. GitHub

Download the package from Github and install it locally:

```bash
git clone https://github.com/chen-yang-yuan/mcDETECT
cd mcDETECT/mcDETECT_package
python3 setup.py install --user
```

## Dependencies and environment setup

### Reproducible Conda environment (recommended)

To ensure full reproducibility of the mcDETECT package and all analyses in this repository, we provide a Conda environment specification file [env.yaml](code/utils/env.yaml). This file records the Python version and all major dependencies used to develop mcDETECT and generate the results in the paper.

You can recreate the environment by running:

```bash
conda env create -f env.yaml
conda activate mcDETECT
```

Using this environment is strongly recommended to avoid version conflicts and to ensure full reproducibility. While mcDETECT may work under other environments or with newer package versions, they are not guaranteed to reproduce all results exactly.

### Core Python dependencies

mcDETECT depends on the following major Python packages: anndata, miniball, numpy, pandas, rtree, scanpy, scipy, shapely, scikit-learn.

The exact versions used in our analyses are specified in `env.yaml`.

### Other environments the software has been tested on

In addition to the provided `env.yaml`, mcDETECT has been tested in the following environments:

Environment 1:

* System: macOS Sequoia 15.3.1 (Apple M2 Max)
* Python: 3.11.4
* Python packages: anndata = 0.9.1, miniball = 1.2.0, numpy = 1.24.3, pandas = 2.0.2, rtree = 1.2.0, scanpy = 1.10.3, scipy = 1.10.1, shapely = 2.0.1, sklearn = 1.2.2

Environment 2:
* System: macOS Sequoia 15.3.1 (Apple M1 Pro)
* Python: 3.9.12
* Python packages: anndata = 0.8.0, miniball = 1.2.0, numpy = 1.23.5, pandas = 2.2.3, rtree = 0.9.7, scanpy = 1.10.3, scipy = 1.13.1, shapely = 2.0.1, sklearn = 1.5.2

## Tutorial

For a step-by-step tutorial on `mcDETECT`, please see [tutorial](tutorial/tutorial.md).<br>

Toy datasets used in this tutorial can be downloaded from [Dropbox](https://www.dropbox.com/scl/fo/gxt64ilg55p44iwj1dox3/AO-LRvZUQnJU9twvtaEdpcY?rlkey=bjk5dv5sqnhinblapr12wtzau&st=owdm92gz&dl=0).

## Contributing

Source code: [mcDETECT_package](mcDETECT_package).<br>

We are continuing adding new features. Bug reports or feature requests are welcome.<br>

Last update: 01/25/2026, version 2.0.15.

## Citation