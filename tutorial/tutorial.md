# mcDETECT Tutorial

Authors: Chenyang Yuan, Krupa Patel, Hongshun Shi, Hsiao-Lin V. Wang, Feng Wang, Ronghua Li, Yangping Li, Victor G. Corces, Hailing Shi, Sulagna Das, Jindan Yu, Peng Jin, Bing Yao* and Jian Hu*

### Outline
1. [Installation](#installation)
2. [Import modules](#import-python-modules)
3. [Read in data](#read-in-data)

### Installation

### Import Python modules


```python
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import scanpy as sc

from model_temp import *

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
```

### Read in data


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


```python
mc = mcDETECT(type = "Xenium", transcripts = transcripts, syn_genes = syn_genes, nc_genes = nc_genes, eps = 1.5, grid_len = 1, cutoff_prob = 0.95, alpha = 5, low_bound = 3,
              size_thr = 5, in_nucleus_thr = (0.5, 0.5), l = 1, rho = 0.2, s = 1, nc_top = 20, nc_thr = 0.1)
```


```python
sphere = mc.detect()
```

    1 out of 12 genes processed!
    2 out of 12 genes processed!
    3 out of 12 genes processed!
    4 out of 12 genes processed!
    5 out of 12 genes processed!
    6 out of 12 genes processed!
    7 out of 12 genes processed!
    8 out of 12 genes processed!
    9 out of 12 genes processed!
    10 out of 12 genes processed!
    11 out of 12 genes processed!
    12 out of 12 genes processed!
    Merging spheres...
    Negative control filtering...



```python
sphere
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
      <th>sphere_x</th>
      <th>sphere_y</th>
      <th>sphere_z</th>
      <th>layer_z</th>
      <th>sphere_r</th>
      <th>size</th>
      <th>comp</th>
      <th>in_nucleus</th>
      <th>gene</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5861.525313</td>
      <td>2021.429797</td>
      <td>15.259961</td>
      <td>15.259961</td>
      <td>1.115372</td>
      <td>8</td>
      <td>3</td>
      <td>0.000000</td>
      <td>Snap25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5823.012341</td>
      <td>2477.027071</td>
      <td>18.744452</td>
      <td>18.744452</td>
      <td>1.400544</td>
      <td>13</td>
      <td>5</td>
      <td>0.000000</td>
      <td>Snap25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5805.578936</td>
      <td>2419.213116</td>
      <td>18.711572</td>
      <td>18.711572</td>
      <td>1.355859</td>
      <td>9</td>
      <td>3</td>
      <td>0.000000</td>
      <td>Snap25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5831.996698</td>
      <td>2545.542771</td>
      <td>18.262820</td>
      <td>18.262820</td>
      <td>1.168398</td>
      <td>7</td>
      <td>2</td>
      <td>0.000000</td>
      <td>Snap25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5800.522809</td>
      <td>2731.226949</td>
      <td>16.212029</td>
      <td>16.212029</td>
      <td>1.322234</td>
      <td>7</td>
      <td>2</td>
      <td>0.000000</td>
      <td>Snap25</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1421</th>
      <td>6519.312500</td>
      <td>2139.578125</td>
      <td>17.562500</td>
      <td>17.562500</td>
      <td>0.908403</td>
      <td>3</td>
      <td>1</td>
      <td>0.000000</td>
      <td>Slc32a1</td>
    </tr>
    <tr>
      <th>1422</th>
      <td>6576.820312</td>
      <td>2377.734375</td>
      <td>14.804688</td>
      <td>14.804688</td>
      <td>1.148358</td>
      <td>5</td>
      <td>1</td>
      <td>0.000000</td>
      <td>Slc32a1</td>
    </tr>
    <tr>
      <th>1423</th>
      <td>6633.195312</td>
      <td>2388.710938</td>
      <td>16.890625</td>
      <td>16.890625</td>
      <td>0.476370</td>
      <td>3</td>
      <td>1</td>
      <td>0.333333</td>
      <td>Slc32a1</td>
    </tr>
    <tr>
      <th>1424</th>
      <td>6530.494635</td>
      <td>2487.223874</td>
      <td>15.567417</td>
      <td>15.567417</td>
      <td>1.246287</td>
      <td>8</td>
      <td>4</td>
      <td>0.000000</td>
      <td>Slc32a1</td>
    </tr>
    <tr>
      <th>1425</th>
      <td>6506.195312</td>
      <td>2733.085938</td>
      <td>14.562500</td>
      <td>14.562500</td>
      <td>0.586406</td>
      <td>3</td>
      <td>1</td>
      <td>0.000000</td>
      <td>Slc32a1</td>
    </tr>
  </tbody>
</table>
<p>1279 rows × 9 columns</p>
</div>




```python
a = mc.profile(sphere)
```
