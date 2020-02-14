Disclaimer:
This software is made publicly for research use only. It is provided WITHOUT ANY WARRANTY.
 
------------------------------------------------------------------------------------------

Citation:
Please cite this paper if you use the provided code:

Mashaan Alshammari, Masahiro Takatsuka,
Approximate spectral clustering with eigenvector selection and self-tuned k,
Pattern Recognition Letters,
Volume 122,
2019,
Pages 31-37,
ISSN 0167-8655,

------------------------------------------------------------------------------------------

How to use:

Run BATCH_Points.m which will execute the following:
	1) PRE_Points.m to load toy data, csv files are the groundtruth labels.
	2) RUN_Points.m to perform spectral clustering with 4 functions to estimate k:
		2.1) CostEigenGap.m a conventional method to estimate k
		2.2) CostZelnik.m uses the method proposed by (Zelnik-manor 2005) to estimate k
		2.3) CostDBIOverLambda.m uses the method proposed by our paper to estimate k
		2.4) CostDBIOverLambdaPCA.m a uses the method proposed by our paper to estimate k followed by PCA variance filtering
    3) POST_Points.m to compute the accuracy of clustering

------------------------------------------------------------------------------------------

Provided by Mashaan Alshammari

mashaan14 at gmail dot com
mashaan dot awad at outlook dot com

July 03, 2019.