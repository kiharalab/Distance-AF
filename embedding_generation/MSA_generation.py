from os.path import join,exists
import os
from tqdm import tqdm
import sys
from utils.creat_dir import create_dir


def msa_generation(hhblits_bin_path, hhsearch_bin_path,db_dir,fasta_path,output_dir):
    #Check path and create necessary directory
    assert exists(hhblits_bin_path), 'The execution path for hhblits is not existed, please follow XXX to install!'
    assert exists(hhsearch_bin_path), 'The execution path for hhsearch is not existed, please follow XXX to install!'
    assert exists(fasta_path), 'The input fasta file is not existed, please use a valid path!'
    
    target_name = os.path.split(fasta_path)[1].split('.')[0]
    create_dir(output_dir)
    # run hhblits and hhsearch
    hhblits_cmd = f'hhblits_bin_path \
                    -i {output_dir}/{target_name}.fasta \
                    -d {db_dir}/uniref/UniRef30_2020_06 \
                    -o {output_dir}/{target_name}_Uniref30.hhr \
                    -oa3m {output_dir}/{target_name}_0.001.a3m \
                    -oalis {output_dir}/{target_name}_0.001 \
                    -n 3 -e 0.001 -id 99 -cov 50 -diff inf -cpu 8 -maxfilt 100000'
    hhsearch_cmd = f'hhsearch_bin_path \
                    -i {output_dir}/{target_name}_0.001.a3m \
                    -d {db_dir}/pdb70/pdb70 \
                    -o {output_dir}/{target_name}.hhr \
                    -Z 2000 -B 2000 -cpu 8'

    os.system(hhblits_cmd)
    os.system(hhsearch_cmd)
