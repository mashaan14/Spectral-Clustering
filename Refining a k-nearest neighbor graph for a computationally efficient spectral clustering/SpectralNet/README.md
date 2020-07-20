## Running Guide
This project is based on the oriignal SpectralNet code which can be found on:<br/>
https://github.com/KlugerLab/SpectralNet

The file \core\pairs.py sets pairs for Siamese net using the value provided in `siam_k`. We included a new parameter called `use_mu0` that if it was set to True it will discard the value of `siam_k` and used our method in Algorithm 1 to set pairs for Siamese net.

### Files that we modified from the original SpectralNet code
- \core\pairs.py
	to implement our method
- \core\data.py
	to upload Aggregation and Compound datasets in csv
- \core\run.py
	to include Aggregation and Compound datasets
## Python Setup
This installation tutorial was prepared on insights from this blog:<br/>
Installing a Python Based Machine Learning Environment in Windows 10<br/>
https://towardsdatascience.com/installing-keras-tensorflow-using-anaconda-for-machine-learning-44ab28ff39cb

- Download and install Anaconda Navigator
- Download and install VS Code
- launch Anaconda Prompt to execute the following tasks:
	- Downgrade Python to a Keras & Tensorflow compatible version:
		- `conda install python=3.6`
	- Create a new CPU conda environment
		- `conda create --name PythonCPU`
	- activate the CPU conda environment
		- `activate PythonCPU`
	- install Keras & Tensorflow CPU versions
		- `conda install -c anaconda keras`
	- Install Spyder IDE
		- `conda install spyder`
	- Install Pandas with read and write excel files
		- `conda install -c anaconda pandas`
		- `conda install -c anaconda xlrd`
		- `conda install -c anaconda xlwt`
	- Install the Seaborn library
		- `conda install -c anaconda seaborn`		
	- To install scikit-learn
		- `conda install -c anaconda scikit-learn`	
	- Install Pillow to handle images
		- `conda install pillow`
	- Install annoy
		- `conda install -c conda-forge python-annoy`
	- Install munkres
		- `conda install -c conda-forge munkres`
- To start working in this enviroment, launch Anaconda Prompt and type:
	- `activate PythonCPU`
	- `spyder`
---
written by Mashaan Alshammari<br/>
mashaan14 at gmail dot com<br/>
mashaan dot awad at outlook dot com<br/>
July 20, 2020.
