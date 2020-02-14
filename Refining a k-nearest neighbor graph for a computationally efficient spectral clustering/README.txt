Disclaimer:
This software is made publicly for research use only. It is provided WITHOUT ANY WARRANTY.
 
------------------------------------------------------------------------------------------

Citation:
Please cite these papers if you use the provided code:

Mashaan Alshammari, Masahiro Takatsuka,
Approximate spectral clustering with eigenvector selection and self-tuned k,
Pattern Recognition Letters,
Volume 122,
2019,
Pages 31-37,
ISSN 0167-8655,

Mashaan Alshammari, Masahiro Takatsuka,
Approximate spectral clustering density-based similarity for noisy datasets,
Pattern Recognition Letters,
Volume 128,
2019,
Pages 155-161,
ISSN 0167-8655,

------------------------------------------------------------------------------------------

How to use:

Run BATCH_Points.m which will execute the following:
	1) PRE_Points.m to load toy data, csv files are the groundtruth labels.
	OPTIONAL: let variable PlotShow = true if you want to see plots
	2) if variable k equals zero means number of clusters is unknown and the algorithm will try to guess it
		2.1) RUN_Points_VQ.m to perform approximate spectral clustering for:
					kmeans approximation	+ local sigma edges
					SOM approximation	+ local sigma edges
					kmeans approximation	+ CONN edges
					SOM approximation	+ CONN edges
					kmeans approximation	+ CONNHybrid edges
					SOM approximation	+ CONNHybrid edges
		2.2) RUN_Points_Fast.m to perform spectral clustering with the proposed refined k-nearest nieghbor
	3) if variable k does not equal zero means number of clusters is known and the algorithm will cluster data to k clusters
		3.1) RUN_Points_VQ.m to perform approximate spectral clustering for:
					kmeans approximation	+ local sigma edges
					SOM approximation	+ local sigma edges
					kmeans approximation	+ CONN edges
					SOM approximation	+ CONN edges
					kmeans approximation	+ CONNHybrid edges
					SOM approximation	+ CONNHybrid edges
		3.2) RUN_Points_Fast.m to perform spectral clustering with the proposed refined k-nearest nieghbor		
    4) POST_Points.m to compute the accuracy and adjusted Rand index of clustering

------------------------------------------------------------------------------------------

Provided by Mashaan Alshammari

mashaan14 at gmail dot com
mashaan dot awad at outlook dot com

November 06, 2019.
