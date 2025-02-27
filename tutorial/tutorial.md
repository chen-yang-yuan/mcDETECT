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

### 2. Import Python modules

Compiling this tutorial file needs the following Python packages:


```python
import anndata
import math
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import miniball
import numpy as np
import pandas as pd
import scanpy as sc
from mcDETECT import mcDETECT

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
```

### 3. Read in data

`mcDETECT` requires the following input:

* Transcript file (data frame)


```python
transcripts = pd.read_parquet("toy_data/transcripts.parquet")
```

We need to rename some columns of the transcript file to combat


```python
transcripts = transcripts[['cell_id', 'overlaps_nucleus', 'feature_name', 'x_location', 'y_location', 'z_location']]
transcripts = transcripts.rename(columns = {"feature_name": "target", "x_location": "global_x", "y_location": "global_y", "z_location": "global_z"})
transcripts.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cell_id</th>
      <th>overlaps_nucleus</th>
      <th>target</th>
      <th>global_x</th>
      <th>global_y</th>
      <th>global_z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>163006771</th>
      <td>fgdhmaei-1</td>
      <td>0</td>
      <td>A1cf</td>
      <td>5994.734375</td>
      <td>2021.468750</td>
      <td>15.125000</td>
    </tr>
    <tr>
      <th>163006772</th>
      <td>UNASSIGNED</td>
      <td>0</td>
      <td>A2m</td>
      <td>5763.109375</td>
      <td>2043.625000</td>
      <td>15.781250</td>
    </tr>
    <tr>
      <th>163006773</th>
      <td>UNASSIGNED</td>
      <td>0</td>
      <td>A2m</td>
      <td>5951.984375</td>
      <td>2085.984375</td>
      <td>16.578125</td>
    </tr>
    <tr>
      <th>163006774</th>
      <td>hieeideh-1</td>
      <td>1</td>
      <td>Aatf</td>
      <td>5757.593750</td>
      <td>2163.453125</td>
      <td>17.281250</td>
    </tr>
    <tr>
      <th>163006775</th>
      <td>fghnlpdi-1</td>
      <td>1</td>
      <td>Aatf</td>
      <td>5969.406250</td>
      <td>2149.406250</td>
      <td>17.625000</td>
    </tr>
  </tbody>
</table>
</div>




```python
transcripts
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cell_id</th>
      <th>overlaps_nucleus</th>
      <th>target</th>
      <th>global_x</th>
      <th>global_y</th>
      <th>global_z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>163006771</th>
      <td>fgdhmaei-1</td>
      <td>0</td>
      <td>A1cf</td>
      <td>5994.734375</td>
      <td>2021.468750</td>
      <td>15.125000</td>
    </tr>
    <tr>
      <th>163006772</th>
      <td>UNASSIGNED</td>
      <td>0</td>
      <td>A2m</td>
      <td>5763.109375</td>
      <td>2043.625000</td>
      <td>15.781250</td>
    </tr>
    <tr>
      <th>163006773</th>
      <td>UNASSIGNED</td>
      <td>0</td>
      <td>A2m</td>
      <td>5951.984375</td>
      <td>2085.984375</td>
      <td>16.578125</td>
    </tr>
    <tr>
      <th>163006774</th>
      <td>hieeideh-1</td>
      <td>1</td>
      <td>Aatf</td>
      <td>5757.593750</td>
      <td>2163.453125</td>
      <td>17.281250</td>
    </tr>
    <tr>
      <th>163006775</th>
      <td>fghnlpdi-1</td>
      <td>1</td>
      <td>Aatf</td>
      <td>5969.406250</td>
      <td>2149.406250</td>
      <td>17.625000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>185456895</th>
      <td>fikpmdpm-1</td>
      <td>0</td>
      <td>Vtn</td>
      <td>6502.234375</td>
      <td>2760.000000</td>
      <td>14.859375</td>
    </tr>
    <tr>
      <th>185456896</th>
      <td>UNASSIGNED</td>
      <td>0</td>
      <td>Wapl</td>
      <td>6731.968750</td>
      <td>2854.625000</td>
      <td>21.890625</td>
    </tr>
    <tr>
      <th>185456897</th>
      <td>fikpmdpm-1</td>
      <td>0</td>
      <td>Wnt4</td>
      <td>6503.468750</td>
      <td>2754.656250</td>
      <td>16.156250</td>
    </tr>
    <tr>
      <th>185456898</th>
      <td>fikpmdpm-1</td>
      <td>1</td>
      <td>Ywhaz</td>
      <td>6501.953125</td>
      <td>2752.484375</td>
      <td>14.609375</td>
    </tr>
    <tr>
      <th>185456899</th>
      <td>fikpmdpm-1</td>
      <td>1</td>
      <td>Zmpste24</td>
      <td>6501.359375</td>
      <td>2752.156250</td>
      <td>14.562500</td>
    </tr>
  </tbody>
</table>
<p>8019192 rows × 6 columns</p>
</div>



* Synaptic markers (list)


```python
syn_genes = ['Snap25', 'Camk2a', 'Slc17a7', 'Vamp2', 'Syp', 'Syn1', 'Dlg4', 'Gria2', 'Gap43', 'Gria1', 'Bsn', 'Slc32a1']
```

* Negative control markers (list)


```python
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
