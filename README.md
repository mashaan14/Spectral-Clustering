# Spectral-Clustering

[![DOI](http://img.shields.io/badge/doi-10.1016/j.patcog.2021.107869-36648B.svg)](https://doi.org/10.1016/j.patcog.2021.107869)
[![Paper](http://img.shields.io/badge/arXiv-2302.11296-b31b1b.svg)](https://arxiv.org/abs/2302.11296)
[![Papers with Code](http://img.shields.io/badge/PaperswithCode-2302.11296-21cbce.svg)](https://paperswithcode.com/paper/refining-a-k-nearest-neighbor-graph-for-a)

## Refining a k-nearest neighbor graph for a computationally efficient spectral clustering
If you find this repository helpful, please cite this paper:
```bibtex
@article{ALSHAMMARI2021107869,
	title = {Refining a k-nearest neighbor graph for a computationally efficient spectral clustering},
	journal = {Pattern Recognition},
	year = {2021},
	volume = {114},
	pages = {107869},	
	doi = {https://doi.org/10.1016/j.patcog.2021.107869},
	author = {Mashaan Alshammari and John Stavrakakis and Masahiro Takatsuka}
}
```

## How to use
Run BATCH_Points.m which will execute the following:
1.	PRE_Points.m to load toy data, csv files are the groundtruth labels.
	- OPTIONAL: let variable PlotShow = true if you want to see plots
2.	if variable k equals zero means number of clusters is unknown and the algorithm will try to guess it
	- RUN_Points_VQ.m to perform approximate spectral clustering for:
		- kmeans approximation	+ local sigma edges
		- SOM approximation	+ local sigma edges
		- kmeans approximation	+ CONN edges
		- SOM approximation	+ CONN edges
		- kmeans approximation	+ CONNHybrid edges
		- SOM approximation	+ CONNHybrid edges
	- RUN_Points_Fast.m to perform spectral clustering with the proposed refined k-nearest nieghbor
3. if variable k does not equal zero means number of clusters is known and the algorithm will cluster data to k clusters
	- RUN_Points_VQ.m to perform approximate spectral clustering for:
		- kmeans approximation	+ local sigma edges
		- SOM approximation	+ local sigma edges
		- kmeans approximation	+ CONN edges
		- SOM approximation	+ CONN edges
		- kmeans approximation	+ CONNHybrid edges
		- SOM approximation	+ CONNHybrid edges
	- RUN_Points_Fast.m to perform spectral clustering with the proposed refined k-nearest nieghbor
4. POST_Points.m to compute the accuracy and adjusted Rand index of clustering

---
written by [Mashaan Alshammari](https://mashaan14.github.io/mashaan/)<br/>
November 06, 2019.
