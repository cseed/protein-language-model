from torch.utils.data import Dataset, DataLoader
import pyfastx
import os
from google.cloud import storage
import numpy as np
import torch

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

token_dic_sim = {
    '<cls>': 0,
    'A': 1,
    'R': 2,
    'N': 3,
    'D': 4,
    '<pad>': 5,
    '<eos>': 6,
    '<mask>': 7,
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

def token_transform_sim(working_seq):
    return token_transform(working_seq, token_dic_sim)

def padded_collate(batch, token_dic):
    # Get the maximum length in the batch
    max_len = max(len(item) for item in batch)
    
    # Pad each array with the pad token
    padded_batch = [np.concatenate((item, np.full(max_len - len(item), token_dic['<pad>']))) for item in batch]

    # Stack the padded arrays to create a batch
    return torch.tensor(np.vstack(padded_batch))

def padded_collate_uniref(batch):
    return padded_collate(batch, token_dic_uniref)

def padded_collate_sim(batch):
    return padded_collate(batch, token_dic_sim)

class SequenceDatasetTokenized(Dataset):
    def __init__(self, fasta_file_name, data_dir, transform = lambda x:x):
        self.transform = transform
        curr_dir = os.getcwd()
        data_dir_path = os.path.join(curr_dir, data_dir)
        fasta_file_path = os.path.join(data_dir_path, fasta_file_name)
        # fasta_file_path = '50seqs.fasta'
        if(not os.path.isfile(fasta_file_path)):
            print('downloading fasta file')
            storage_client = storage.Client()
            bucket = storage_client.bucket('uniref_2018_data')
            blob = bucket.blob('uniref50_2018_sax.fasta.bgz')
            if(not os.path.isdir(data_dir_path)):
                os.mkdir(data_dir_path)
            blob.download_to_filename(fasta_file_path)
        
        # train_sequences, test_sequences = train_test_split(data, test_size=0.2, random_state=42)
        self.data = pyfastx.Fasta(fasta_file_path)

    def __len__(self):
        # Get the number of sequences in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        if(self.transform is None):
            return self.data[idx].seq
        else:
            return self.transform(self.data[idx].seq)
        
class Uniref50DatasetTokenized(SequenceDatasetTokenized):
    def __init__(self, transform = None):
        fasta_file_name = 'uniref50_2018_sax.fasta.bgz'
        data_dir = 'uniref50_2018'
        super().__init__(fasta_file_name, data_dir, transform)

class SimulatedDataSet1(SequenceDatasetTokenized):
    def __init__(self, transform = None):
        fasta_file_name = 'sim1.fasta'
        data_dir = 'simulated_data'
        super().__init__(fasta_file_name, data_dir, transform)
        



 
