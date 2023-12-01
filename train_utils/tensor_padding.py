import torch

def pad_tensor(batch, lens, dims, key, dtype='float'):
	padded_tensor = torch.zeros(dims)
	if dtype == 'long':
		padded_tensor = torch.zeros(dims).long()
	for i_batch, (data, length) in enumerate(zip(batch, lens)):
		i_data = data[key]
		if len(dims) == 4 and key == 'mt_msa':
			padded_tensor[i_batch, :, :length, :] = i_data
		elif len(dims) == 4:
			padded_tensor[i_batch, :, :length, :length] = i_data
		elif len(dims) == 3 and key == 'embed':
			padded_tensor[i_batch, :, :length] = i_data
		elif len(dims) == 3:
			padded_tensor[i_batch, :length, :length] = i_data
		elif len(dims) == 2:
			padded_tensor[i_batch, :length] = i_data
	return padded_tensor

def pad_tensor2(batch, lens, seq_lens,dims, key, dtype='float'):
	padded_tensor = torch.zeros(dims)
	if dtype == 'long':
		padded_tensor = torch.zeros(dims).long()
	#print('padded tensor: ', padded_tensor.size())
	for i_batch, (data, length, seq_length) in enumerate(zip(batch, lens, seq_lens)):
		i_data = data[key]
		#print('seq_length: ', seq_length)
		padded_tensor[i_batch, :seq_length, :length, :] = i_data
	return padded_tensor

def pad_tensor3(batch, lens, dims, key, dtype='float'):
	padded_tensor = torch.zeros(dims)
	if key == 'aatype':
		padded_tensor.fill_(20)
	if dtype == 'long':
		padded_tensor = torch.zeros(dims).long()
	for i_batch, (data, length) in enumerate(zip(batch, lens)):
		#print(key)
		i_data = data[key]
		if len(dims) == 4 and key == 'coords':
			padded_tensor[i_batch, :length, :, :] = i_data
		elif len(dims) == 4:
			padded_tensor[i_batch, :length, :length, :] = i_data
		elif len(dims) == 3 and (key == 'embed' or key == 'msa_mask'):
			padded_tensor[i_batch, :, :length] = i_data
		elif len(dims) == 3:
			padded_tensor[i_batch, :length, :] = i_data
		elif len(dims) == 2:
			if key == 'mask':
				padded_tensor[i_batch, :length] = i_data
			else:
				padded_tensor[i_batch, :i_data.size(0)] = i_data
	return padded_tensor

def pad_tensor4(batch, lens, dims, key, dtype='float'):
	padded_tensor = torch.zeros(dims)
	if dtype == 'long':
		padded_tensor = torch.zeros(dims).long()
	for i_batch, (data, length) in enumerate(zip(batch, lens)):
		i_data = data[key]
		if len(dims) == 4:
			padded_tensor[i_batch, :i_data.size(0), :, :] = i_data
		else:
			padded_tensor[i_batch, :i_data.size(0), :] = i_data
	return padded_tensor