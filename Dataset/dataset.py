from torch.utils.data.dataset import Dataset
import numpy as np
import torch
import os
import random
from utils.read_pdb_info import read_pdb_info, get_seq
from protein_utils import protein, all_atom, data_transforms

class DistAF_Dataset(Dataset):

    def __init__(self, args=None):
        self.train_file = args.target_file

        with open(self.train_file, 'r') as f:
            self.targets = f.read().splitlines()
    
        self.max_len = args.max_len
        self.embedding_file = args.emd_file
        self.fasta_file = args.fasta_file
        self.initial_pdb = args.initial_pdb
        self.window_info = args.window_info
        self.dist_constraint_file = args.dist_info
        self.output_dir = args.output_dir
        self.args = args
        self.target_seq_len = 0
        self.start_position = 0
        self.end_position = self.start_position + self.target_seq_len

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        target = self.targets[index]
        target_seq = get_seq(self.fasta_file)
        self.target_seq_len = len(target_seq)
        self.output_dir = os.path.join(self.output_dir,target)
        self.end_position = self.target_seq_len

        data = {}
        data["target"] = target
        
        if self.embedding_file.endswith('.npz'):
            emd = np.load(self.embedding_file)
        else:
            import pickle
            with open(self.embedding_file, 'rb') as f:
                emd = pickle.load(f)['representations']
        pair = emd['pair']
        single = emd['single']
    
        single = torch.tensor(single)
        pair = torch.tensor(pair)

        resolution = 1
 
        data['resolution'] = torch.tensor([resolution])

        initial_pdb = self.initial_pdb
        coords = read_pdb_info(initial_pdb,self.target_seq_len)
        mask = np.ones(self.target_seq_len)
        
        data['single_representation'] = single
        single_emd = single.unsqueeze(0)
        pair_emd = pair.unsqueeze(0)
   
        data['single_representation'] = single_emd
        data['embed'] = pair_emd

        data['coords'] = coords
        data['mask'] = mask
 
        '''all atom attribute load using DeepMind's utils'''
        
        pdb_str = ''
        with open(initial_pdb,"r") as f:
            pdb_str = f.read()
        prot = protein.from_pdb_string(pdb_str)
       
        prot_dict = {
            'aatype': np.asarray(prot.aatype),
            'all_atom_positions':np.asarray(prot.atom_positions),
            'all_atom_mask':np.asarray(prot.atom_mask),
        }						
        prot_dict = all_atom.make_atom14_positions(prot_dict)
        data['aatype'] = torch.tensor(np.asarray(prot.aatype))
        for key in prot_dict.keys():
            data[key] = torch.tensor(prot_dict[key][:, ...])
        
        protein_object = protein.from_pdb_string(pdb_str)
        residue_index = torch.tensor(protein_object.residue_index)
        protein_object = data_transforms.atom37_to_frames(
                                {'aatype': torch.tensor(protein_object.aatype),
                                'all_atom_positions': torch.tensor(protein_object.atom_positions),  # (..., 37, 3)
                                'all_atom_mask': torch.tensor(protein_object.atom_mask)})
        protein_object = data_transforms.atom37_to_torsion_angles(protein_object)
        protein_object.update({'residue_index': residue_index})
        protein_object.update({'residue_index': residue_index,
                                'chi_angles_sin_cos': protein_object["torsion_angles_sin_cos"][..., 3:, :],
                                'chi_mask': protein_object["torsion_angles_mask"][..., 3:],
                                'seq_mask':  torch.ones(protein_object["aatype"].shape, dtype=torch.float32)})
        for key in protein_object.keys():
            data[key] = protein_object[key][:, ...]

        data['seq_length'] = self.target_seq_len
        
        data['aatype_start'] = 0
        
        # variables needed in distance af defined below
        if os.path.exists(os.path.join(self.output_dir, f"{target}_constraint.pt")):
            data['domain_window'] = torch.load(os.path.join(self.output_dir, f"{target}_domain_window.pt"))
            data['dist_constraint'] = torch.load(os.path.join(self.output_dir, f"{target}_constraint.pt"))
        else:
            data['domain_window'] = None
            data['dist_constraint'] = None
            domain_window = torch.zeros((self.target_seq_len,self.target_seq_len))
            with open(self.window_info, 'r') as file:
                lines = file.readlines()
            for line in lines:
                line = line.strip()
                start = int(line.split(',')[0])-1
                end = int(line.split(',')[1])+1
                domain_window[start:end,start:end] = 1
            data['domain_window'] = domain_window

            dist_constraint = torch.zeros((self.target_seq_len,self.target_seq_len))
            with open(self.dist_constraint_file, 'r') as file:
                lines = file.readlines()
            for line in lines:
                line = line.strip()
                first_resi = int(line.split(',')[0])-1
                second_resi = int(line.split(',')[1])-1
                dist_cons = float(line.split(',')[2])
                dist_constraint[first_resi, second_resi] = dist_cons
                dist_constraint[second_resi, first_resi] = dist_cons

            data['dist_constraint'] = dist_constraint
            torch.save(data['dist_constraint'], os.path.join(self.output_dir, f"{target}_constraint.pt"))
            torch.save(data['domain_window'], os.path.join(self.output_dir, f"{target}_domain_window.pt")) 

        return data