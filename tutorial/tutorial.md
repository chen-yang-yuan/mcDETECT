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

### 3. Read in data

`mcDETECT` requires the following input:

* Transcript file (data frame)

We need to rename some columns of the transcript file to combat




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



### 4. Parameter settings

### 5. Synapse detection

### 6. Spatial domain assignment

### 7. Synapse transcriptome profiling

### 8. Synapse subtyping
