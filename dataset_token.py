from torch.utils.data import Dataset, DataLoader
import pyfastx
import os
from google.cloud import storage
import numpy as np
import torch
import time
import logging

# Suppress logs from Google Cloud Storage client
logging.getLogger('google').setLevel(logging.WARNING)
logging.getLogger('google.cloud').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('google.auth').setLevel(logging.WARNING)

token_dic_uniref = {
    '<cls>': 0,
    'A': 1,
    'R': 2,
    'N': 3,
    'D': 4,
    'C': 5,
    'Q': 6,
    'E': 7,
    'G': 8,
    'H': 9,
    'I': 10,
    'L': 11,
    'K': 12,
    'M': 13,
    'F': 14,
    'P': 15,
    'S': 16,
    'T': 17,
    'W': 18,
    'Y': 19,
    'V': 20,
    'X': 21, # Unknown not sure if the token should be <Unk>
    '<pad>': 22,
    '<eos>': 23,
    'B':24,
    'Z':25,
    'U':26,
    '<mask>': 27,
    'O': 28
}



def token_transform(working_seq, token_dic):
    seq_list = list(working_seq)
    tokenized = [token_dic[char] for char in seq_list]
    final_rep = [token_dic['<cls>']] + tokenized + [token_dic['<eos>']]
    if(len(final_rep) <= 1024):
        return np.array(final_rep)
    else:
        start_index = np.random.randint(0,len(final_rep))
        return np.array(final_rep[start_index:start_index+1024])

def token_transform_uniref(working_seq):
    return token_transform(working_seq, token_dic_uniref)

def padded_collate(batch, token_dic):
    # Get the maximum length in the batch
    max_len = max(len(item) for item in batch)
    
    # Pad each array with the pad token
    padded_batch = [np.concatenate((item, np.full(max_len - len(item), token_dic['<pad>']))) for item in batch]

    # Stack the padded arrays to create a batch
    return torch.tensor(np.vstack(padded_batch))

def padded_collate_uniref(batch):
    return padded_collate(batch, token_dic_uniref)

class SequenceDatasetTokenized(Dataset):
    def __init__(self, data_file_name, data_index_name, data_dir, transform = lambda x:x):
        self.transform = transform
        curr_dir = os.getcwd()
        data_dir_path = os.path.join(curr_dir, data_dir)
        data_file_path = os.path.join(data_dir_path, data_file_name)
        data_index_path = os.path.join(data_dir_path, data_index_name)

        if not os.path.isfile(data_file_path):
            print('downloading data file')
            start_time = time.time()
            storage_client = storage.Client()
            bucket = storage_client.bucket('uniref_2018_data')
            blob = bucket.blob('all_uniref_concatenated.txt')
            if(not os.path.isdir(data_dir_path)):
                os.mkdir(data_dir_path)
            blob.download_to_filename(data_file_path)
            end_time = time.time()
            print(f'Downloaded seq data in {end_time - start_time} sec')

        if not os.path.isfile(data_index_path):
            print('downloading index file')
            start_time = time.time()
            storage_client = storage.Client()
            bucket = storage_client.bucket('uniref_2018_data')
            blob = bucket.blob('sequence_index.npy')
            if(not os.path.isdir(data_dir_path)):
                os.mkdir(data_dir_path)
            blob.download_to_filename(data_index_path)
            end_time = time.time()
            print(f'Downloaded index in {end_time - start_time} sec')

        
        # train_sequences, test_sequences = train_test_split(data, test_size=0.2, random_state=42)
        self.data_path = data_file_path
        self.data_index = np.load(data_index_path)
        self.file_pool = []

    def __len__(self):
        # Get the number of sequences in the dataset
        return len(self.data_index)

    def __getitem__(self, idx):
        try:
            f = self.file_pool.pop()
        except:
            # binary so read returns bytes object
            f = open(self.data_path, 'r')

        offset = self.data_index[idx,0]
        length = self.data_index[idx,1]
        f.seek(offset, 0)
        seq = f.read(length)
        assert len(seq) == length

        # return the file to the pool
        self.file_pool.append(f)

        # convert bytes object to string
        #seq = seq.decode('utf-8')

        if(self.transform is None):
            return seq
        else:
            return self.transform(seq)
        
class Uniref50DatasetTokenized(SequenceDatasetTokenized):
    def __init__(self, transform = None):
        #fasta_file_name = 'uniref50_2018_sax.fasta.bgz'
        data_file_name = 'all_uniref_concatenated.txt'
        data_index_name = 'sequence_index.npy'
        data_dir = 'uniref50_2018'
        super().__init__(data_file_name, data_index_name, data_dir, transform)

"""
dataset = Uniref50DatasetTokenized(transform=token_transform_uniref)
loader = DataLoader(dataset=dataset, batch_size=1000, shuffle=True, collate_fn=padded_collate_uniref, num_workers=4)

dataiter = iter(loader)

res1 = next(dataiter)
res2 = next(dataiter)

print(res1[0][0])
print(res1[1][0])
print(res2[0][0])
print(res2[1][0])
"""


 
