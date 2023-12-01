# Distance-AF

<a href="https://github.com/marktext/marktext/releases/latest">
   <img src="https://img.shields.io/badge/DistanceAF-v1.0.0-green">
   <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20-green">
   <img src="https://img.shields.io/badge/Language-python3-green">
   <img src="https://img.shields.io/badge/dependencies-tested-green">
   <img src="https://img.shields.io/badge/licence-GNU-green">
</a>  

Distance-AF is a computational tool using deep learning to predict protein structure with distance constraints between residues.  

Copyright (C) 2023 Yuanyuan Zhang, Zicong Zhang, Yuki Kagaya, Genki Terashi, Daisuke Kihara, and Purdue University. 

License: GPL v3. (If you are interested in a different license, for example, for commercial use, please contact us.) 

Contact: Daisuke Kihara (dkihara@purdue.edu)

For technical problems or questions, please reach to Yuanyuan Zhang (zhang038@purdue.edu).

## Installation
<details>

### System Requirements
CPU: >=8 cores <br>
Memory (RAM): >=50Gb. For fasta sequence than 3,000 residues, memory space should be higher than 200GB if the sequence is provided. <br>
GPU: any GPU supports CUDA with at least 12GB memory. <br>
GPU is required for Distance-AF and no CPU version is available for Distance-AF since it is too slow.

## Pre-required software
### Required 
Python 3 : https://www.python.org/downloads/     
### Optional
Pymol (for map visualization): https://pymol.org/2/    
Chimera (for map visualization): https://www.cgl.ucsf.edu/chimera/download.html  

## Environment set up  
### 1. [`Install git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
### 2. Clone the repository in your computer 
```
git clone  https://github.itap.purdue.edu/kiharalab/Distance-AF.git && cd Distance-AF
```
### 3. Build dependencies.   
You have two options to install dependency on your computer:
#### 3.2 Install with anaconda (Recommended)
##### 3.2.1 [`install anaconda`](https://www.anaconda.com/download). 
##### 3.2.2 Install dependency in command line
Make sure you are in the CryoREAD directory and then run 
```
conda env create -f environment.yml
```
Each time when you want to run this software, simply activate the environment by
```
conda activate dist-af
conda deactivate(If you want to exit) 
```



## Usage
   
```
python3 main.py --devide_id=1
```

