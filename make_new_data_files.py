import numpy as np
import pyfastx

fa = pyfastx.Fasta('uniref50_2018/uniref50_2018_sax.fasta.bgz')

index_mat = np.empty((len(fa), 2), dtype=np.int64)
i = 0
offset = 0
with open('all_uniref_concatenated.txt','w') as f:
    for item in fa:
        seq = item.seq
        if i % 100 == 0:
            print(i)
        curr_seq_len = len(seq)
        f.write(seq)
        index_mat[i, 0] = offset
        index_mat[i, 1] = curr_seq_len
        offset = offset + curr_seq_len
        i += 1
f.close()

np.save('sequence_index.npy', index_mat)
    
