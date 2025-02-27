# mcDETECT Tutorial

Authors: Chenyang Yuan, Krupa Patel, Hongshun Shi, Hsiao-Lin V. Wang, Feng Wang, Ronghua Li, Yangping Li, Victor G. Corces, Hailing Shi, Sulagna Das, Jindan Yu, Peng Jin, Bing Yao* and Jian Hu*

### Outline

1. [Installation](#1-installation)
2. [Import Python modules](#2-import-python-modules)
3. [Read in data](#3-read-in-data)
4. [Parameter settings](#4-parameter-settings)
5. [Synapse detection](#5-synapse-detection)
6. [Spatial domain assignment](#6-spatial-domain-assignment)
7. [Synapse transcriptome profiling](#7-synapse-transcriptome-profiling)
8. [Synapse subtyping](#8-synapse-subtyping)

### 1. Installation

The detailed installation procedure can be found in [Installation](../README.md/#installation). Here I directly install the package by running:

```bash
python3 -m pip install mcDETECT
```


```python
import anndata
import math
import miniball
import numpy as np
import pandas as pd
import scanpy as sc
```

### 2. Import Python modules

Besides required Python packages for `mcDETECT` listed in [Dependencies](README.md), this tutorial also needs the following packages:


```python
import matplotlib.colors as clr
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
```

### 3. Read in data


```python
# Read and format transcripts
transcripts = pd.read_parquet("toy_data/transcripts.parquet")
transcripts = transcripts[['cell_id', 'overlaps_nucleus', 'feature_name', 'x_location', 'y_location', 'z_location']]
transcripts = transcripts.rename(columns = {"feature_name": "target", "x_location": "global_x", "y_location": "global_y", "z_location": "global_z"})

# Define synaptic markers
syn_genes = ['Snap25', 'Camk2a', 'Slc17a7', 'Vamp2', 'Syp', 'Syn1', 'Dlg4', 'Gria2', 'Gap43', 'Gria1', 'Bsn', 'Slc32a1']

# Read negative control markers
nc_genes = pd.read_csv('toy_data/negative_controls.csv')
nc_genes = list(nc_genes['Gene'])
```

### 4. Parameter settings


```python
mc = mcDETECT(type = "Xenium", transcripts = transcripts, syn_genes = syn_genes, nc_genes = nc_genes, eps = 1.5, grid_len = 1, cutoff_prob = 0.95, alpha = 5, low_bound = 3,
              size_thr = 5, in_nucleus_thr = (0.5, 0.5), l = 1, rho = 0.2, s = 1, nc_top = 20, nc_thr = 0.1)
```

### 5. Synapse detection


```python
sphere = mc.detect()
```


```python
sphere
```


```python
a, b = mc.construct_grid()
```


```python
len(b)
```


```python
aaa = mc.spot_expression(grid_len=50)
```


```python

```

### 6. Spatial domain assignment

### 7. Synapse transcriptome profiling


```python
a = mc.profile(sphere)
```

### 8. Synapse subtyping
