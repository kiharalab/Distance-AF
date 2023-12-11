## Distance-AF embedding generation
### 1. Google colab(suggested)
#### Running instructions
 1. Go to [Distance_AF_embedding.ipynb](https://github.com/kiharalab/Distance-AF/blob/main/Distance_AF_embedding.ipynb).
 2. At the right top, click 'Open in Colab'.
 3. Install the environment and dependency by running the first cell '*1. Install third-party software*' and second cell '*2. Download AlphaFold*'.
 4. **Prepare your sequence and job name** at '*3. Enter the amino acid sequence(s) to fold*'.
 	 + **REQUIRED**: filled out your sequence at the text box **sequence_1**.
	 + **OPTIONAL**: change the job_name text box with any name you like.
	 + **OTHERS**: keep empty for sequence 2, sequence 3, etc for current version.
 5. Run the thrid cell '*3. Enter the amino acid sequence(s) to fold'*, fourth cell '*4. Search against genetic databases*' and fifth cell '5. Run AlphaFold and download embeddings' consecutively.
 6. The embedding zip file will be downloaded automatically named in 'job_name.zip'
#### More details
 
 7. In the zip file, it contains `model_1.npz`,  `model_2.npz`,  `model_3.npz`,  `model_4.npz`,  `model_5.npz`, corresponding to the embedding file derived from 5 models from AlphaFold. The PDB files `model_1.pdb`, `model_2.pdb`, `model_3.pdb`, `model_4.pdb`, `model_5.pdb` are the AlphaFold2 predicted structure.
 8. To choose more appropriate embedding file among the 5 npz files, you can compare your initial structure with the 5 PDB files, choose the corresponding npz file which is closest to your initial structure. For example, you find `model_2.pdb` is the closest one, you then choose `model_2.npz` as the embedding file.
### 2. Run Embedding generation locally(on going...)
