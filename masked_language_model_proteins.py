import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, functional
from torch.utils.data import DataLoader, Subset

from dataset_token import (
    Uniref50DatasetTokenized,
    SimulatedDataSet1,
    token_dic_sim,
    token_dic_uniref,
    token_transform_sim,
    padded_collate_sim,
    token_transform_uniref,
    padded_collate_uniref,
)
import time
import numpy as np
import gzip

import matplotlib.pyplot as plt
import argparse
import torch.optim.lr_scheduler as lr_scheduler
from functools import partial
import logging

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, max_length: int, embed_dim: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(ntoken, embed_dim, padding_idx = args.token_dic['<pad>'])
        self.pos_embedding = LearnedPositionalEmbedding(
            num_embeddings=max_length,
            embedding_dim=embed_dim,
            padding_idx=args.token_dic['<pad>'],
        )
        self.norm = nn.LayerNorm(embed_dim)
        encoder_layers = TransformerEncoderLayer(embed_dim, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.linear = nn.Linear(embed_dim, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        self.embedding.weight.data.normal_(mean=0.0, std=0.02)
        self.linear.bias.data.zero_()
        self.linear.weight.data.normal_(mean=0.0, std=0.02)


    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        padding_mask = src == args.token_dic['<pad>']
        src = self.embedding(src) * math.sqrt(self.embed_dim) + self.pos_embedding(src)
        output = self.transformer_encoder(src, src_key_padding_mask=padding_mask)
        output = self.linear(output) #could try replacing this with inverse (i.e. transpose) of the embedding layer
        return output

class LearnedPositionalEmbedding(nn.Embedding):
    """
    Copied from ESM2

    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor):
        """Input is expected to be of size [bsz x seqlen]."""
        if input.size(1) > self.max_positions:
            raise ValueError(
                f"Sequence length {input.size(1)} above maximum "
                f" sequence length of {self.max_positions}"
            )
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return functional.embedding(
            positions,
            self.weight,
            self.padding_idx,
            # self.max_norm,
            # self.norm_type,
            # self.scale_grad_by_freq,
            # self.sparse,
        )

def generate_data_mask(src_tensor, p=0.15):
    mask = torch.rand_like(src_tensor, dtype=torch.float) < p
    mask[
        (src_tensor == args.token_dic['<cls>']) |\
        (src_tensor == args.token_dic['<eos>']) |\
        (src_tensor == args.token_dic['<pad>'])
    ] = False
    true_indices = torch.nonzero(mask, as_tuple=False)
    return true_indices


def train_one_epoch(model: nn.Module, train_dataset, args) -> None:
    model.train()  # turn on train mode

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=args.padded_collate,
        shuffle=True
    )
    train_data = iter(train_loader)
    
    if args.n_train_batches == 0:
        n_batches = len(train_loader)
    else:
        n_batches = args.n_train_batches


    print(f'num training batches available: {len(train_loader)}')
    print(f'num batches we will run: {n_batches}')

    xvals = []
    yvals = []
    total_loss = 0.
    start_time = time.time()
    
    for batch in range(n_batches):

        data = next(train_data)
        data_mask_indices = generate_data_mask(data, p=args.dropout)
        masked_data = data.clone()
        masked_data[data_mask_indices[:,0], data_mask_indices[:,1]] = args.token_dic['<mask>']
        output = model(masked_data.to('cuda'))
        output_at_mask_positions = output[data_mask_indices[:,0], data_mask_indices[:,1]]
        targets_at_mask_positions = data[data_mask_indices[:,0], data_mask_indices[:,1]].to(output_at_mask_positions.device)
        loss = args.loss_function(output_at_mask_positions, targets_at_mask_positions)
        total_loss += loss.item()

        args.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        args.optimizer.step()

        xvals.append(batch) # we should use weights and biases or tensorboard instead
        yvals.append(float(loss))
        
        if batch % args.log_interval == 0 and batch > 0:
            per_batch_logging(batch, start_time, total_loss, n_batches, args)
            total_loss = 0
            start_time = time.time()
        args.scheduler.step()

    return xvals, yvals

def evaluate(model: nn.Module, eval_dataset, args) -> float:
    model.eval()  # turn on evaluation mode
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        collate_fn=args.padded_collate,
        shuffle=True
    )
    eval_data = iter(eval_loader)

    if args.n_eval_batches == 0:
        n_batches = len(eval_loader)
    else:
        n_batches = args.n_eval_batches
    
    total_loss = 0.
    total_seq_len = 0
    with torch.no_grad():
        for i in range(n_batches):
            data = next(eval_data)
            data_mask_indices = generate_data_mask(data, p=0.15)
            masked_data = data.clone()
            masked_data[data_mask_indices[:,0], data_mask_indices[:,1]] = args.token_dic['<mask>']
            output = model(masked_data.to('cuda'))
            output_at_mask_positions = output[data_mask_indices[:,0], data_mask_indices[:,1]]
            targets_at_mask_positions = data[data_mask_indices[:,0], data_mask_indices[:,1]]
            targets_at_mask_positions = targets_at_mask_positions.to(output_at_mask_positions.device)
            loss = args.loss_function(output_at_mask_positions, targets_at_mask_positions)
            seq_len = targets_at_mask_positions.size(0)
            total_loss += seq_len * loss.item()
            total_seq_len += seq_len
    total_avg_loss = total_loss / total_seq_len
    ppl = math.exp(total_avg_loss)

    return total_avg_loss, ppl

def per_batch_logging(batch, start_time, total_loss, n_batches, args):

    batch_lr = args.scheduler.get_last_lr()[0]
    ms_per_batch = (time.time() - start_time) * 1000 / args.log_interval
    cur_loss = total_loss / args.log_interval
    ppl = math.exp(cur_loss)

    print(f'{batch:5d}/{n_batches:5d} batches | '
            f'lr {batch_lr:.5g} | ms/batch {ms_per_batch:5.2f} | '
            f'loss {cur_loss:5.2f} | ppl {ppl:8.2f} | '
    )
    logging.info(f'{batch:5d}/{n_batches:5d} batches | '
            f'lr {batch_lr:.5g} | ms/batch {ms_per_batch:5.2f} | '
            f'loss {cur_loss:5.2f} | ppl {ppl:8.2f} | ')

def per_epoch_logging(epoch, elapsed, val_loss, val_ppl, xvals, yvals, print_graph):
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
        f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    logging.info('-' * 89)
    logging.info(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
        f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    logging.info('-' * 89)


    if print_graph:
        plt.scatter(xvals, yvals)
        plt.title('Training Loss vs Batch')
        plt.xlabel('Batch')
        plt.ylabel('Training File')
        plt.savefig(f'loss_graph{epoch}.png')

def end_of_training_logging(test_loss, test_ppl):
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f} | '
        f'test ppl {test_ppl:8.2f}')
    print('=' * 89)

def full_training(model, train_dataset, validation_dataset, args):
    best_val_loss = float('inf')

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        for epoch in range(1, args.n_epochs + 1):
            epoch_start_time = time.time()
            xvals, yvals = train_one_epoch(model, train_dataset, args)
 
            val_loss, val_ppl = evaluate(model, validation_dataset, args)
            elapsed = time.time() - epoch_start_time
            per_epoch_logging(epoch, elapsed, val_loss, val_ppl, xvals, yvals, args.print_graph)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)

        model.load_state_dict(torch.load(best_model_params_path)) # load best model states
    return model

def get_data():
    # Load train, test, val splits
    """
    with gzip.open('train_indices.npy.gz', 'rb') as f:
        train_indices = np.load(f)
    """
    train_indices = np.load("train_indices.npy")
    val_indices = np.load('val_indices.npy')
    test_indices = np.load('test_indices.npy')

    train_dataset = Subset(Uniref50DatasetTokenized(transform=token_transform_uniref), train_indices)
    validation_dataset = Subset(Uniref50DatasetTokenized(transform=token_transform_uniref), val_indices)
    test_dataset = Subset(Uniref50DatasetTokenized(transform=token_transform_uniref), test_indices)

    print(f'Num train samples: {len(train_dataset)}')
    print(f'Num val samples: {len(validation_dataset)}')
    print(f'Num test samples: {len(test_dataset)}')

    ntokens = len(args.token_dic)

    return train_dataset, validation_dataset, test_dataset, ntokens

def get_simulated_data():
    train_dataset = SimulatedDataSet1(transform=token_transform_sim)
    validation_dataset = SimulatedDataSet1(transform=token_transform_sim)
    test_dataset = SimulatedDataSet1(transform=token_transform_sim)
    ntokens = len(args.token_dic)
    return train_dataset, validation_dataset, test_dataset, ntokens

def _get_inverse_sqrt_schedule_lr_lambda(current_step: int, *, num_warmup_steps: int, timescale: int = None):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    shift = timescale - num_warmup_steps
    decay = 1.0 / math.sqrt((current_step + shift) / timescale)
    return decay

def main(args):
    # can also make the masking/corruption match esms
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    logging.basicConfig(filename='transformer_run.log', filemode='w',format='%(message)s')
    logging.root.setLevel(logging.DEBUG)
    logging.info(args)

    if args.sim:
        train_dataset, validation_dataset, test_dataset, n_tokens = get_simulated_data()
    else: train_dataset, validation_dataset, test_dataset, n_tokens = get_data()

    model = TransformerModel(
        n_tokens,
        args.max_length,
        args.embed_dim,
        args.n_head,
        args.d_hid,
        args.n_layers,
        args.dropout
    ).to(dtype=torch.bfloat16,device=device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    args.loss_function = nn.CrossEntropyLoss()
    num_warmup_steps = args.num_warmup_steps
    timescale = num_warmup_steps
    lr_lambda = partial(
        _get_inverse_sqrt_schedule_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        timescale=timescale
    )
    
    args.optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), lr=args.lr, weight_decay = 0.01)
    # ESM2 paper using linear decay to 1/10 original lr after 2000 warmup steps
    args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda, last_epoch=-1)
    # args.scheduler = torch.optim.lr_scheduler.StepLR(args.optimizer, 1.0, gamma=0.95) #to do
    model = full_training(model, train_dataset, validation_dataset, args)
    
    test_loss, test_ppl = evaluate(model, test_dataset, args)
    end_of_training_logging(test_loss, test_ppl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Description of your script.")
    
    parser.add_argument('--embed-dim', type=int, default=10)
    parser.add_argument('--d-hid', type=int, default=40)
    parser.add_argument('--n-head', type=int, default=2)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--n-train-batches', type=int, default=100)
    parser.add_argument('--n-eval-batches', type=int, default=10)
    parser.add_argument('--n-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--print-graph', action='store_true')
    parser.add_argument('--num-warmup-steps', type=int, default=40)
    parser.add_argument('--sim', action='store_true')

    args = parser.parse_args()

    if args.sim:
        args.token_dic = token_dic_sim
        args.padded_collate = padded_collate_sim
        args.max_length = 32
    else:
        args.token_dic = token_dic_uniref
        args.padded_collate = padded_collate_uniref
        args.max_length = 1024

    main(args)

