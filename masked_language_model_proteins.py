import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, functional
from torch.utils.data import DataLoader, Subset

from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import Mlp
import torch.nn.functional as F

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
import json
from datetime import datetime
import subprocess
import threading
import xml.etree.ElementTree as ET
from xmljson import badgerfish as bf
import math

class JsonLinesHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)

    def emit(self, record):
        log_entry = self.format(record)
        self.stream.write(f'{log_entry}\n')
        self.flush()

class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps(record.getMessage())

# Configure the logger
log_filename = 'transformer_run.jsonl'
json_handler = JsonLinesHandler(log_filename, mode='w')
json_formatter = JsonFormatter()

logging.root.setLevel(logging.DEBUG)
logging.root.addHandler(json_handler)

# Log nvidia-smi info on gpu and memory utilization
def fetch_gpu_info():
    try:
        # Run the nvidia-smi command and get the output in XML format
        result = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True)
        xml_output = result.stdout
        # Parse the XML output
        root = ET.fromstring(xml_output)
        # Extract memory utilization and gpu utilization
        gpu_utilizations = []
        for gpu in root.findall('gpu'):
            utilization = gpu.find('utilization')
            memory_util = utilization.find('memory_util').text
            gpu_util = utilization.find('gpu_util').text
            gpu_utilizations.append({
                'memory_utilization': memory_util,
                'gpu_utilization': gpu_util
            })
        # Create the final JSON object
        output = {
            'type': 'nvidia-smi-output',
            'gpu_utilizations': gpu_utilizations
        }
        # Log the JSON output
        logging.info(output)
    except Exception as e:
        logging.error(f"Failed to fetch GPU info: {e}")

def monitor_gpu(interval=1):
    while True:
        fetch_gpu_info()
        time.sleep(interval)

def monitor_cpu_load(interval=1):
    while True:
        with open('/proc/loadavg','r') as f:
            val = f.readlines()
            log_info = {'type': 'cpu-load', 'cpu-usage': val}
            logging.info(log_info)
        f.close()
        time.sleep(interval)

# From ESM Github https://github.com/facebookresearch/esm/blob/main/esm/modules.py#L298
def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class LMHead(nn.Module):
    """Head for masked language modeling. Adapted from ESM https://github.com/facebookresearch/esm/blob/main/esm/modules.py#L298"""

    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, output_dim)


    def forward(self, features):
        x = self.dense(features)
        y = gelu(x)
        z = self.layer_norm(y)
        xx = self.linear(z)
        return xx

# Code taken from https://gist.github.com/kklemon/98e491ff877c497668c715541f1bf478
class FlashAttentionTransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        nlayers,
        nhead,
        d_hid,
        dropout,
        norm_first=False,
        activation=F.gelu
    ):
        super().__init__()

        self._pad_input = pad_input
        self._unpad_input = unpad_input
        
        
        mixer_cls = partial(
            MHA,
            num_heads=nhead,
            use_flash_attn=True,
            rotary_emb_dim=0
        )

        mlp_cls = partial(Mlp, hidden_features=d_hid)
     
        self.layers = nn.ModuleList([
            Block(
                embed_dim,
                mixer_cls=mixer_cls,
                mlp_cls=mlp_cls,
                resid_dropout1=dropout,
                resid_dropout2=dropout,
                prenorm=norm_first,
            ) for _ in range(nlayers)
        ])
    
    def forward(self, x, src_key_padding_mask):
        batch, seqlen = x.shape[:2]

        x, indices, cu_seqlens, max_seqlen_in_batch = self._unpad_input(x, ~src_key_padding_mask)
        
        for layer in self.layers:
            x = layer(x, mixer_kwargs=dict(
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen_in_batch
            ))

        x = self._pad_input(x, indices, batch, seqlen)
            
        return x



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
        """
        self.norm = nn.LayerNorm(embed_dim)
        encoder_layers = TransformerEncoderLayer(embed_dim, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        """
        self.transformer_encoder = FlashAttentionTransformerEncoder(embed_dim, nlayers, nhead, d_hid, dropout)
        self.linear = LMHead(embed_dim, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'weight' in name:
                    nn.init.normal_(param, mean=0.0, std=0.02)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)


    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[batch_size, seq_len, ntoken]``
        """
        padding_mask = src == args.token_dic['<pad>']
        
        src = self.embedding(src) * math.sqrt(self.embed_dim) + self.pos_embedding(src)
        
        output = self.transformer_encoder(src, src_key_padding_mask=padding_mask)
        output = self.linear(output)
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


def train_one_epoch(model: nn.Module, train_dataset, epoch, args) -> None:
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

    start_time = time.time()
    
    # Zero grad at the start of each epoch
    args.optimizer.zero_grad()

    for batch in range(n_batches):

        data = next(train_data)
        data_mask_indices = generate_data_mask(data)
        masked_data = data.clone()
        masked_data[data_mask_indices[:,0], data_mask_indices[:,1]] = args.token_dic['<mask>']
        output = model(masked_data.to('cuda'))
        output_at_mask_positions = output[data_mask_indices[:,0], data_mask_indices[:,1]]
        targets_at_mask_positions = data[data_mask_indices[:,0], data_mask_indices[:,1]].to(output_at_mask_positions.device)
        loss = args.loss_function(output_at_mask_positions, targets_at_mask_positions)
        
        curr_batch_lr = args.scheduler.get_last_lr()[0]
        curr_batch_loss = loss.item()
        
        # Scale loss for accumulation TODO: this is what online said is it right?
        loss = loss / args.batches_per_update
        loss.backward()

        if (batch + 1) % args.batches_per_update == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            args.optimizer.step()
            args.scheduler.step()
            args.optimizer.zero_grad()

        if batch % args.log_interval == 0:
            per_batch_logging(batch, start_time, curr_batch_loss, n_batches, epoch, curr_batch_lr, args)
            start_time = time.time()

def evaluate(model: nn.Module, eval_dataset, epoch, args) -> float:
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

    print(f'num eval batches available: {len(eval_loader)}')
    print(f'num batches we will run: {n_batches}')
    
    total_loss = 0.
    total_seq_len = 0
    start_time = time.time()

    with torch.no_grad():
        for i in range(n_batches):
            data = next(eval_data)
            data_mask_indices = generate_data_mask(data)
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

            if i % args.log_interval == 0 and i > 0:
                per_batch_eval_logging(i, start_time, loss, n_batches, epoch, args)
                start_time = time.time()

    total_avg_loss = total_loss / total_seq_len
    ppl = math.exp(total_avg_loss)

    return total_avg_loss, ppl

def per_batch_logging(batch, start_time, total_loss, n_batches, epoch, batch_lr, args):
    ms_per_batch = (time.time() - start_time) * 1000 / args.log_interval
    cur_loss = total_loss / args.log_interval
    ppl = math.exp(cur_loss)

    print(f'{batch:5d}/{n_batches:5d} batches | '
            f'lr {batch_lr:.5g} | ms/batch {ms_per_batch:5.2f} | '
            f'loss {cur_loss:5.2f} | ppl {ppl:8.2f} | '
    )
    log_args = {'type': 'training-batch','batch_number': batch, 'total_batches': n_batches, 'learning_rate': batch_lr, 'batch_time': ms_per_batch, 'loss': cur_loss, 'ppl': ppl, 'epoch': epoch}
    logging.info(log_args)

def per_batch_eval_logging(batch, start_time, loss, n_batches, epoch, args):
    ms_per_batch = (time.time() - start_time) * 1000 / args.log_interval
    ppl = math.exp(loss)
    log_args = {'type': 'eval-batch','batch_number': batch, 'total_batches': n_batches, 'batch_time': ms_per_batch, 'loss': loss, 'ppl': ppl, 'epoch': epoch}
    logging.info(log_args)

def per_epoch_logging(epoch, elapsed, val_loss, val_ppl):
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
        f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)
    
    log_args = {'type': 'end-of-epoch', 'epoch': epoch, 'total_time': elapsed, 'val_loss': val_loss, 'val_ppl': val_ppl}
    logging.info(log_args)

def end_of_training_logging(test_loss, test_ppl):
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f} | '
        f'test ppl {test_ppl:8.2f}')
    print('=' * 89)
    log_args = {'type': 'end-of-training', 'test_loss': test_loss, 'test_ppl': test_ppl}
    logging.info(log_args)

def full_training(model, train_dataset, validation_dataset, args):
    best_val_loss = float('inf')

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        for epoch in range(1, args.n_epochs + 1):
            epoch_start_time = time.time()
            train_one_epoch(model, train_dataset, epoch, args)
 
            val_loss, val_ppl = evaluate(model, validation_dataset, epoch, args)
            elapsed = time.time() - epoch_start_time
            per_epoch_logging(epoch, elapsed, val_loss, val_ppl)

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
    
    # Log the training parameters
    param_info = {'type': 'training-params'}
    for key, value in vars(args).items():
        param_info[key] = value
    logging.info(param_info)

    train_dataset, validation_dataset, test_dataset, n_tokens = get_data()
    
    # Train the model
    model = TransformerModel(
        n_tokens,
        args.max_length,
        args.embed_dim,
        args.n_head,
        args.d_hid,
        args.n_layers,
        args.dropout
    ).to(dtype=torch.bfloat16,device=device)
    
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Name: {name}, Shape: {param.shape}, Number of parameters: {param.numel()}")
            total_params += param.numel()
    print(f"Total number of parameters: {total_params}")

    tokens_per_batch = args.batch_size * args.max_length
    args.batches_per_update = max(math.ceil(args.tokens_per_update / tokens_per_batch), 1)
    print(f'will update gradient every {args.batches_per_update} batches')

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
    
    test_loss, test_ppl = evaluate(model, test_dataset, args.n_epochs, args)
    end_of_training_logging(test_loss, test_ppl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Description of your script.")
    
    parser.add_argument('--embed-dim', type=int, default=320)
    parser.add_argument('--d-hid', type=int, default=1280)
    parser.add_argument('--n-head', type=int, default=20)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--n-train-batches', type=int, default=100)
    parser.add_argument('--n-eval-batches', type=int, default=10)
    parser.add_argument('--n-epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--num-warmup-steps', type=int, default=40)
    parser.add_argument('--tokens-per-update', type=int, default = 2000000)
    parser.add_argument('--max-length', type=int, default = 1024)


    args = parser.parse_args()

    args.token_dic = token_dic_uniref
    args.padded_collate = padded_collate_uniref

    monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
    monitor_thread.start()

    moniter_thread2 = threading.Thread(target=monitor_cpu_load, daemon=True)
    moniter_thread2.start()

    main(args)

