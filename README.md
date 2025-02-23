# mcDETECT

## mcDETECT: Decoding 3D Spatial Synaptic Transcriptomes with Subcellular-Resolution Spatial Transcriptomics

#### Chenyang Yuan, Krupa Patel, Hongshun Shi, Hsiao-Lin V. Wang, Feng Wang, Ronghua Li, Yangping Li, Victor G. Corces, Hailing Shi, Sulagna Das, Jindan Yu, Peng Jin, Bing Yao* and Jian Hu*

mcDETECT is a computational framework designed to identify and profile individual synapses using *in situ* spatial transcriptomics (iST) data. It starts by examining the subcellular distribution of synaptic mRNAs in an iST sample. Unlike cell-type specific marker genes, which are typically found within nuclei, mRNAs of synaptic markers often form small aggregations outside the nuclei. mcDETECT uses a density-based clustering approach to identify these extranuclear aggregations. This involves calculating the Euclidean distance between mRNA points and defining the neighborhood of each point within a specified search radius. Points are then categorized into core points, border points, and noise points based on their reachability from neighboring points. mcDETECT recognizes each bundle of core and border points as a synaptic aggregation. To minimize false positives, it excludes aggregations that significantly overlap with nuclei identified by DAPI staining. Subsequently, mcDETECT repeats this process for multiple synaptic markers, merging aggregations from different markers with high overlaps. After encompassing all markers, an additional filtering step is performed to remove aggregations that contain mRNAs from negative control genes, which are known to be enriched only in nuclei. The remaining aggregations are considered individual synaptic aggregations. mcDETECT then uses the minimum enclosing sphere of each aggregation to gather all mRNA molecules and summarizes their counts for all measured genes to define the spatial transcriptome profile of individual synapses.

![mcDETECT workflow](docs/workflow.png)<br>

For more details, see our [manuscript]().<br>