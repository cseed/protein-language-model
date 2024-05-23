import numpy as np
import os
import pyfastx

fasta_file_name = 'uniref50_2018_sax.fasta.bgz'
data_dir = 'uniref50_2018'
curr_dir = os.getcwd()
data_dir_path = os.path.join(curr_dir, data_dir)
fasta_file_path = os.path.join(data_dir_path, fasta_file_name)

data = pyfastx.Fasta(fasta_file_path)

print(f'total number of sequences: {len(data)}')

test_size = int(len(data) * 0.2)

val_size = int(len(data)*0.8*0.15)

indices = np.arange(0,len(data))
np.random.shuffle(indices)

test_indices = indices[0:test_size]
val_indices = indices[test_size: test_size+val_size]
train_indices = indices[test_size+val_size:]

np.save('test_indices.npy', test_indices)

np.save('val_indices.npy', val_indices)

np.save('train_indices.npy', train_indices)

print(f'total number of test sequences: {len(test_indices)}')

print(f'total number of val sequences: {len(val_indices)}')

print(f'total number of train sequences: {len(train_indices)}')


