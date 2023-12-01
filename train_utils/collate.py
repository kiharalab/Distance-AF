from train_utils.tensor_padding import pad_tensor, pad_tensor2, pad_tensor3, pad_tensor4
import torch

def collate_fn(batch, args):

    gt_keys_3d = ['all_atom_positions', 'atom14_gt_positions', 'atom14_alt_gt_positions']
    gt_keys_2d = ['all_atom_mask', 'atom14_atom_exists', 'atom14_gt_exists', 'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 'atom37_atom_exists','atom14_alt_gt_exists', 'atom14_atom_is_ambiguous']
    gt_frames_keys_3d = ['rigidgroups_gt_exists', 'rigidgroups_group_exists', 'rigidgroups_group_is_ambiguous']
    gt_frames_keys_4d = ['torsion_angles_sin_cos', 'alt_torsion_angles_sin_cos']
    gt_frames_keys_5d = ['rigidgroups_gt_frames', 'rigidgroups_alt_gt_frames']
    collate_dict = {}

    keys_3d = ['dist', 'mu', 'rho', 'theta', 'sce', 'no']
    keys_2d = ['phi', 'psi']

    lens = [data['embed'].shape[1] for data in batch]

    embed_size = batch[0]['embed'].shape[-1]
    batch_size = len(batch)
    max_len = max(lens)
    single_size = batch[0]['single_representation'].shape[-1]

    for key in keys_3d:
        if key in batch[0]: #It wont be present for test dataloader
            collate_dict[key] = pad_tensor(batch, lens, [batch_size, max_len, max_len], key, dtype='long')
    for key in keys_2d:
        if key in batch[0]:
            collate_dict[key] = pad_tensor(batch, lens, [batch_size, max_len], key, dtype='long')

    if args.embed =='msa_transformer':
        collate_dict['embed'] = pad_tensor3(batch, lens, [batch_size, max_len, max_len, embed_size], 'embed')
        collate_dict['single_representation'] = pad_tensor3(batch, lens, [batch_size, max_len, single_size], 'single_representation')
        collate_dict['aatype'] = pad_tensor3(batch, lens, [batch_size, max_len], 'aatype', dtype='long')
        collate_dict['residue_index'] = pad_tensor3(batch, lens, [batch_size, max_len], 'residue_index', dtype='long')
        
        for key in gt_keys_3d:
            if 'atom14' in key:
                collate_dict[key] = pad_tensor4(batch, lens, [batch_size, max_len, 14, 3], key)
            else:
                collate_dict[key] = pad_tensor4(batch, lens, [batch_size, max_len, 37, 3], key)
        for key in gt_keys_2d:
            if key == 'residx_atom37_to_atom14':
                collate_dict[key] = pad_tensor4(batch, lens, [batch_size, max_len, 37], key)#, dtype='long')
                continue
            if key == 'residx_atom14_to_atom37':
                collate_dict[key] = pad_tensor4(batch, lens, [batch_size, max_len, 14], key, dtype='long')
                continue
            if 'atom14' in key:
                collate_dict[key] = pad_tensor4(batch, lens, [batch_size, max_len, 14], key)
            else:
                collate_dict[key] = pad_tensor4(batch, lens, [batch_size, max_len, 37], key)
        for key in gt_frames_keys_3d:
            collate_dict[key] = pad_tensor4(batch, lens, [batch_size, max_len, 8], key)
        for key in gt_frames_keys_4d:
            collate_dict[key] = pad_tensor4(batch, lens, [batch_size, max_len, 7, 2], key)
        for key in gt_frames_keys_5d:
            collate_dict[key] = pad_tensor4(batch, lens, [batch_size, max_len, 8, 4, 4], key)
        collate_dict['torsion_angles_mask'] = pad_tensor4(batch, lens, [batch_size, max_len, 7], 'torsion_angles_mask')
        collate_dict['chi_angles_sin_cos'] = pad_tensor4(batch, lens, [batch_size, max_len, 4, 2], 'chi_angles_sin_cos')
        collate_dict['chi_mask'] = pad_tensor4(batch, lens, [batch_size, max_len, 4], 'chi_mask')
        collate_dict['seq_mask'] = pad_tensor3(batch, lens, [batch_size, max_len], 'seq_mask')

    targets = []
    record_lines = []
    resolution = []
    seq_length = list()

    for i_batch, (data, length) in enumerate(zip(batch, lens)):
        target = data["target"]
        res = data['resolution']
        targets.append(target)
        resolution.append(res)
        # record_lines.append(data['record_lines'])
        seq_length.append(data['seq_length'])
    collate_dict['resolution'] = torch.stack(resolution)
    collate_dict['seq_length'] = torch.tensor(seq_length)
    collate_dict['domain_window'] = data['domain_window']
    collate_dict['dist_constraint'] = data['dist_constraint']
    if args.dist:
        collate_dict['aatype_start'] = data['aatype_start']
    return collate_dict, targets
