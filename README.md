

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
## Citation:

Yuanyuan  Zhang, Zicong  Zhang, Yuki  Kagaya, Genki  Terashi, Bowen  Zhao, Yi  Xiong & Daisuke  Kihara. Distance-AF: Modifying Predicted Protein Structure Models by Alphafold2 with User-Specified Distance Constraints. bioRxiv 2023.12.01.569498; doi:  [https://doi.org/10.1101/2023.12.01.569498](https://doi.org/10.1101/2023.12.01.569498)

```
@article {Zhang2023.12.01.569498,
	author = {Yuanyuan Zhang and Zicong Zhang and Yuki Kagaya and Genki Terashi and Bowen Zhao and Yi Xiong and Daisuke Kihara},
	title = {Distance-AF: Modifying Predicted Protein Structure Models by Alphafold2 with User-Specified Distance Constraints},
	elocation-id = {2023.12.01.569498},
	year = {2023},
	doi = {10.1101/2023.12.01.569498},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {The three-dimensional structure of a protein plays a fundamental role in determining its function and has an essential impact on understanding biological processes. Despite significant progress in protein structure prediction, such as AlphaFold2, challenges remain on those hard targets that Alphafold2 does not often perform well due to the complex folding of protein and a large number of possible conformations. Here we present a modified version of the AlphaFold2, called Distance-AF, which aims to improve the performance of AlphaFold2 by including distance constraints as input information. Distance-AF uses AlphaFold2{\textquoteright}s predicted structure as a starting point and incorporates distance constraints between amino acids to adjust folding of the protein structure until it meets the constraints. Distance-AF can correct the domain orientation on challenging targets, leading to more accurate structures with a lower root mean square deviation (RMSD). The ability of Distance-AF is also useful in fitting protein structures into cryo-electron microscopy maps.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2023/12/04/2023.12.01.569498},
	eprint = {https://www.biorxiv.org/content/early/2023/12/04/2023.12.01.569498.full.pdf},
	journal = {bioRxiv}
} 
```


## Installation

### System Requirements
CPU: >=8 cores <br>
Memory (RAM): >=50Gb. For fasta sequence than 3,000 residues, memory space should be higher than 200GB if the sequence is provided. <br>
GPU: any GPU supports CUDA with at least 12GB memory. <br>
GPU is required for Distance-AF and no CPU version is available for Distance-AF since it is too slow.

## Pre-required software
### Required 
Python 3 : https://www.python.org/downloads/     
### Optional
Pymol (for protein structure visualization): https://pymol.org/2/    
Chimera (for map visualization if running on EMD related target): https://www.cgl.ucsf.edu/chimera/download.html  

## Environment set up  
### 1. [`Install git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
### 2. Clone the repository in your computer 
```
git clone  https://github.com/kiharalab/Distance-AF && cd Distance-AF
```
### 3. Build dependencies.   
You have two options to install dependency on your computer:
#### 3.2 Install with anaconda (Recommended)
##### 3.2.1 [`install anaconda`](https://www.anaconda.com/download). 
##### 3.2.2 Install dependency in command line
Make sure you are in the Distance-AF directory and then run 
```
conda env create -f environment.yml
```
Each time when you want to run this software, simply activate the environment by
```
conda activate dist-af
conda deactivate(If you want to exit) 
```



## Usage


### 1. Command parameters
```
usage: main.py [-h] [--target_file=TARGET_FILE] [--emd_file=EMD_FILE] [--dist_info=DIST_INFO] [--window_info=WINDOW_INFO] [--initial_pdb=INITIAL_PDB] [--fasta_file=FASTA_FILE] [--output_dir=OUTPUT_DIR] [--epochs=EPOCHS] [--device_id=DEVICE_ID] [--loose_dist=LOOSE_DIST] [--dist_weight=DIST_WEIGHT]

required arguments:
  -h, --help               show this help message and exit
  --target_file            Target file path, default="Example/1IXCA/1IXCA"
  --emd_file               Embedding file path, npz file, default="Example/1IXCA/model_1.npz"
  --dist_info              Distance constraint file, default="Example/1IXCA/dist_constraint.txt"
  --window_info            Domain window file, default="Example/1IXCA/window.txt"
  --initial_pdb            The PDB file of structure you want to run Distance-AF as start point, default="Example/1IXCA/1IXCA_pred_full.pdb"
  --fasta_file             The fasta file of your sequence, default="Example/1IXCA/1IXCA.fasta"
  --output_dir             The output directory you want to save results, default="./example_output".
  --epochs                 The overfitting iterations, default value: 10000
  --device_id              The GPU id you want to run Distance-AF
  --loose_dist             If loosing the weight of distance_loss near final epochs, default value:1
  --dist_weight            The weight for distance loss, default value: 0.5
```
### 2. Run Distance-AF with user specified target
#### 2.1 Prepare Input files
 + Target file: text file, for example: `1IXCA`
 + 1IXCA
 + Fasta file: fasta format file, for example: `1IXCA.fasta`
 +  \>1IXC chain A
MEFRQLKYFIAVAEAGNMAAAAKRLHVSQPPITRQMQALEADLGVVLLERSHRGIELTAAGHAFLEDARRILELAGRSGDRSRAAARGDVGELSVAYFGTPIYRSLPLLLRAFLTSTPTATVSLTHMTKDEQVEGLLAGTIHVGFSRFFPRHPGIEIVNIAQEDLYLAVHRSQSGKFGKTCKLADLRAVELTLFPRGGRPSFADEVIGLFKHAGIEPRIARVVEDATAALALTMAGAASSIVPASVAAIRWPDIAFARIVGTRVKVPISCIFRKEKQPPILARFVEHVRRSAKD

 + User specified distance constraint file: text format , we recommend users to specify as many constraints you have to achieve better performance. In the following example, we use 6 pairs of distance constraints.
 + `15,221,35.3`: the distance constraint between CA atom of resi 15 and CA atom of resi 221 is 35.3 Å.
 + `17,232,36.9`: the distance constraint between CA atom of resi 17 and CA atom of resi 232 is 36.9 Å.
 + `45,150,34.3`: the distance constraint between CA atom of resi 45 and CA atom of resi 150 is 34.3 Å.
 + `55,190,45.8`: the distance constraint between CA atom of resi 55 and CA atom of resi 190 is 45.8 Å.
 + `65,266,36.0`: the distance constraint between CA atom of resi 65 and CA atom of resi 266 is 36.0 Å.
 + `79,126,22.8`: the distance constraint between CA atom of resi 79 and CA atom of resi 126 is 22.8 Å.
 + Initial predicted file in PDB with full length: PDB format, for example: `1IXCA_pred_full.pdb`
 + We recommend to use the predicted structure file which violates the given distance constraints as initial structure.
 + Domain window info file: text format to specify distince domains. To achieve domain oriented movement.
 + 1,87: resi 1 to resi 87 belong to the first domain.
 + 92,294: resi 92 to resi 294 belong to the second domain.
 + Embedding file: npz format . The output file after the evoformer layer in [AlphaFold2](https://github.com/google-deepmind/alphafold). We will provide further instructions about how to obtain embedding file.
#### 2.2 Command line
    python3 main.py [--target_file=TARGET_FILE] [--emd_file=EMD_FILE] [--dist_info=DIST_INFO] [--window_info=WINDOW_INFO] [--initial_pdb=INITIAL_PDB] [--fasta_file=FASTA_FILE] [--output_dir=OUTPUT_DIR] [--epochs=EPOCHS] [--device_id=DEVICE_ID] [--loose_dist=LOOSE_DIST] [--dist_weight=DIST_WEIGHT]

 + [target_file] is the path of the target file.  
 + [emd_file] is the embedding file, formatted in npz, for your own target.  
 + [dist_info] is the text file with the distance constraints you want to applied.  
 + [window_info] is the text file specifying residues belonging to individual domains.  
 + [initial_pdb] is the initial structure you want to optimize by Distance-AF.  
 + [fasta_file] is the fasta file for the target.  
 + [output_dir] is the parent directory of the output dir you want to save  the results.  
 + [epochs] is the running epochs.  
 + [device_id] specifies the gpu used for running Distance-AF.  
 + [loose_dist] specifies if loosing distance loss weight when distance constraints are roughly satisfied near ending epoch.  
 + [dist_weight] specifies the weight you want to distance loss, larger value, stricter penalty on distance violation.  
#### 2.3 Example command

    python3 main.py --target_file=Example/1IXCA/1IXCA --emd_file=Example/1IXCA/model_1.npz --dist_info=Example/1IXCA/dist_constraint.txt --window_info=Example/1IXCA/window.txt --initial_pdb=Example/1IXCA/1IXCA_pred_full.pdb --fasta_file=Example/1IXCA/1IXCA.fasta --output_dir=./example_output --model_dir=./model_dir --dist_weight=0.5 --loose_dist=1 --device_id=1

   
