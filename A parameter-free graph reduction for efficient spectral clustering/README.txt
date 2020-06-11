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
	2) the variable k is the number of clusters, the algorithm will cluster data to k clusters
		2.1) RUN_Points_VQ.m to perform approximate spectral clustering with parameter $\m$ for:
					kmeans approximation	+ local sigma edges
					kmeans approximation	+ CONN edges
					SOM approximation		+ CONN edges
					kmeans approximation	+ CONNHybrid edges
					SOM approximation		+ CONNHybrid edges
		2.2) RUN_Points_Fast_Old.m to perform spectral clustering with the refined k-nearest nieghbor with parameter $\mu_0$
		2.3) RUN_Points_Fast.m to perform spectral clustering with the proposed method		
    4) POST_Points.m to compute the accuracy and adjusted Rand index of clustering

------------------------------------------------------------------------------------------

Provided by Mashaan Alshammari

mashaan14 at gmail dot com
mashaan dot awad at outlook dot com

April 04, 2020.