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
    token_dic_uniref,
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
import yaml
import pandas as pd
from hail.utils import hadoop_copy

class JsonLinesHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)

    def emit(self, record):
        log_entry = self.format(record)
        if log_entry:
            self.stream.write(f'{log_entry}\n')
            self.flush()

class JsonFormatter(logging.Formatter):
    def format(self, record):
        try:
            return json.dumps(eval(record.getMessage()))
        except Exception:
            return None

# Configure the logger
log_filename = 'transformer_run.jsonl'
json_handler = JsonLinesHandler(log_filename, mode='w')
json_formatter = JsonFormatter()

logging.root.setLevel(logging.DEBUG)
logging.root.addHandler(json_handler)
json_handler.setFormatter(json_formatter)

# Log nvidia-smi info on gpu and memory utilization
def fetch_gpu_info():
    try:
        # Run the nvidia-smi command and get the output in XML format
        result = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True)
        xml_output = result.stdout
        # Parse the XML output
        root = ET.fromstring(xml_output)
        attached_gpus = int(root.find('attached_gpus').text)

        gpu_utilizations = []
        for gpu in root.findall('gpu'):
            product_name = gpu.find('product_name').text

            utilization = gpu.find('utilization')
            memory_util = utilization.find('memory_util').text
            gpu_util = utilization.find('gpu_util').text

            # Extract fb_memory_usage fields
            fb_memory_usage = gpu.find('fb_memory_usage')
            total_memory = fb_memory_usage.find('total').text
            reserved_memory = fb_memory_usage.find('reserved').text
            used_memory = fb_memory_usage.find('used').text
            free_memory = fb_memory_usage.find('free').text

            memory_util_float = float(memory_util.replace('%', '')) / 100.0
            gpu_util_float = float(gpu_util.replace('%', '')) / 100.0

            gpu_utilizations.append({
                'gpu_type': product_name,
                'memory_utilization': memory_util_float,
                'gpu_utilization': gpu_util_float,
                'memory_usage': {
                    'total': total_memory,
                    'reserved': reserved_memory,
                    'used': used_memory,
                    'free': free_memory
                }
            })
        # Create the final JSON object
        output = {
            'type': 'nvidia-smi-output',
            'timestamp': time.time(),
            'attached_gpus': attached_gpus,
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
            log_info = {'type': 'cpu-load', 'timestamp': time.time(), 'cpu-usage': val}
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
        shuffle=True,
        num_workers=4
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
        shuffle=True,
        num_workers=4
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
                per_batch_eval_logging(i, start_time, loss.item(), n_batches, epoch, args)
                start_time = time.time()

    total_avg_loss = total_loss / total_seq_len
    ppl = math.exp(total_avg_loss)

    return total_avg_loss, ppl

def per_batch_logging(batch, start_time, total_loss, n_batches, epoch, batch_lr, args):
    s_per_batch = (time.time() - start_time) / args.log_interval
    cur_loss = total_loss / args.log_interval
    ppl = math.exp(cur_loss)

    print(f'{batch:5d}/{n_batches:5d} batches | '
            f'lr {batch_lr:.5g} | s/batch {s_per_batch:5.2f} | '
            f'loss {cur_loss:5.2f} | ppl {ppl:8.2f} | '
    )
    log_args = {'type': 'training-batch','timestamp': time.time(), 'batch_number': batch, 'total_batches': n_batches, 'learning_rate': batch_lr, 'batch_time_sec': s_per_batch, 'loss': cur_loss, 'ppl': ppl, 'epoch': epoch}
    logging.info(log_args)

def per_batch_eval_logging(batch, start_time, loss, n_batches, epoch, args):
    s_per_batch = time.time() - start_time
    ppl = math.exp(loss)
    log_args = {'type': 'eval-batch','timestamp': time.time(), 'batch_number': batch, 'total_batches': n_batches, 'batch_time_sec': s_per_batch, 'loss': loss, 'ppl': ppl, 'epoch': epoch}
    logging.info(log_args)

def per_epoch_logging(epoch, elapsed, val_loss, val_ppl):
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
        f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)
    
    log_args = {'type': 'end-of-epoch','timestamp': time.time(),'epoch': epoch, 'total_time_sec': elapsed, 'val_loss': val_loss, 'val_ppl': val_ppl}
    logging.info(log_args)

def end_of_training_logging(test_loss, test_ppl):
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f} | '
        f'test ppl {test_ppl:8.2f}')
    print('=' * 89)
    log_args = {'type': 'end-of-training', 'timestamp': time.time(),'test_loss': test_loss, 'test_ppl': test_ppl}
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


def _get_linear_decay_schedule_lr_lambda(current_step: int, *, num_warmup_steps: int, peak_lr: float, total_steps: int):
    if current_step <= num_warmup_steps:
        return float(current_step) / num_warmup_steps
    else:
        decay_start_step = num_warmup_steps
        decay_steps = total_steps - num_warmup_steps
        decay_rate = (1/10 - 1) / decay_steps
        return 1 + decay_rate * (current_step - decay_start_step)

def make_loss_plot(events, run_id):
    end_of_epoch_events = [e for e in events if e['type'] == 'end-of-epoch']
    df_epoch = pd.DataFrame(end_of_epoch_events)
    validation_losses = df_epoch['val_loss'].values

    training_events = [e for e in events if e['type'] == 'training-batch']
    df_train = pd.DataFrame(training_events)
    training_losses = df_train['loss'].values
    
    train_batches_per_epoch = len(df_train) // len(df_epoch)

    epochs = [i*train_batches_per_epoch for i in range(1,len(validation_losses)+1)]

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(training_losses)), training_losses, label='Training Loss')
    plt.plot(epochs, validation_losses, marker='o', linestyle='-', label='Validation Loss')

    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.savefig('loss_plot.png')
    hadoop_copy('loss_plot.png',f'gs://missense-scoring/experiments/{run_id}/loss_plot.png')
    
def make_ppl_plot(events, run_id):
    end_of_epoch_events = [e for e in events if e['type'] == 'end-of-epoch']
    df_epoch = pd.DataFrame(end_of_epoch_events)
    validation_ppl = df_epoch['val_ppl'].values

    training_events = [e for e in events if e['type'] == 'training-batch']
    df_train = pd.DataFrame(training_events)
    training_ppl = df_train['ppl'].values
    
    train_batches_per_epoch = len(df_train) // len(df_epoch)

    epochs = [i*train_batches_per_epoch for i in range(1,len(validation_ppl)+1)]

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(training_ppl)), training_ppl, label='Training Perplexity')
    plt.plot(epochs, validation_ppl, marker='o', linestyle='-', label='Validation Perplexity')

    plt.xlabel('Training Steps')
    plt.ylabel('Perplexity')
    plt.title('Training and Validation Perplexity')
    plt.legend()

    plt.savefig('ppl_plot.png')
    hadoop_copy('ppl_plot.png',f'gs://missense-scoring/experiments/{run_id}/ppl_plot.png')

def make_epoch_summary_table(events, run_id):
    end_of_epoch_events = [e for e in events if e['type'] == 'end-of-epoch']
    df = pd.DataFrame(end_of_epoch_events)
    df.to_csv('end_of_epoch_table.csv')
    hadoop_copy('end_of_epoch_table.csv',f'gs://missense-scoring/experiments/{run_id}/end_of_epoch_table.csv')
    
def make_lr_plot(events, run_id):
    training_events = [e for e in events if e['type'] == 'training-batch']
    df = pd.DataFrame(training_events)
    plt.plot(np.arange(len(df)), df['learning_rate'], label='learning_rate')
    plt.xlabel('batch')
    plt.ylabel('learning_rate')
    plt.savefit('lr_plot.png')
    hadoop_copy('lr_plot.png',f'gs://missense-scoring/experiments/{run_id}/lr_plot.png')

def make_cpu_usage_plots(events, run_id):
    cpu_events = [e for e in events if e['type'] == 'cpu-load']

    cpu_events = [{**e, 'cpu-usage': float(e['cpu-usage'][0].split()[0])} for e in cpu_events]

    df = pd.DataFrame(cpu_events)
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()
    plt.plot(df['timestamp'], df['cpu-usage'], label='cpu-usage')
    plt.xlabel('time (ms)')
    plt.ylabel('cpu-usage')
    plt.savefig('cpu_usage.png')
    hadoop_copy('cpu_usage.png',f'gs://missense-scoring/experiments/{run_id}/cpu_usage.png')

def make_gpu_util_plots(events, run_id):
    gpu_events = [e for e in events if e['type'] == 'nvidia-smi-output']

    new_gpu_events = []
    for e in gpu_events:
        # copy e before modifying it
        gpu_utilizations = e['gpu_utilizations']
        e = dict(e)
        del e['gpu_utilizations']

        for i, util in enumerate(gpu_utilizations):
            e[f'gpu{i}_memory_utilization'] = util['memory_utilization']
            e[f'gpu{i}_utilization'] = util['gpu_utilization']
            e[f'gpu{i}_total_memory'] = int(util['memory_usage']['total'][:-4])
            e[f'gpu{i}_used_memory'] = int(util['memory_usage']['used'][:-4])

        new_gpu_events.append(e)

    gpu_events = new_gpu_events

    df = pd.DataFrame(gpu_events)
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()

    plt.plot(df['timestamp'], df['gpu0_memory_utilization'], label='memory utilization')
    plt.plot(df['timestamp'], df['gpu0_utilization'], label='GPU utilization')
    plt.ylim(bottom=0)
    plt.xlabel('time (s)')
    plt.ylabel('utilization')
    plt.legend()
    plt.savefig('gpu_util.png')
    hadoop_copy('gpu_util.png',f'gs://missense-scoring/experiments/{run_id}/gpu_util.png')
    
    plt.plot(df['timestamp'], df['gpu0_total_memory'], label='total')
    plt.plot(df['timestamp'], df['gpu0_used_memory'], label='used')
    plt.xlabel('time (s)')
    plt.ylabel('memory (MiB)')
    plt.ylim(0)
    plt.legend()
    plt.savefig('memory_util.png')
    hadoop_copy('memory_util.png',f'gs://missense-scoring/experiments/{run_id}/memory_util.png')

def main(args):
    # can also make the masking/corruption match esms
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Log the training parameters
    param_info = {'type': 'training-params'}
    for key, value in vars(args).items():
        if key != 'padded_collate':
            param_info[key] = value
    param_info['timestamp'] = time.time()
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
        _get_linear_decay_schedule_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        peak_lr = args.lr,
        total_steps=270000
    )
    
    args.optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), lr=args.lr, weight_decay = 0.01)
    # ESM2 paper using linear decay to 1/10 original lr after 2000 warmup steps
    args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda, last_epoch=-1)
    # args.scheduler = torch.optim.lr_scheduler.StepLR(args.optimizer, 1.0, gamma=0.95) #to do
    model = full_training(model, train_dataset, validation_dataset, args)
    
    test_loss, test_ppl = evaluate(model, test_dataset, args.n_epochs, args)
    end_of_training_logging(test_loss, test_ppl)

    events = []
    run_id = args.run_id
    with open('transformer_run.jsonl', 'r') as f:
        for line in f:
            events.append(json.loads(line))
    make_loss_plot(events, run_id)
    make_ppl_plot(events, run_id)
    make_epoch_summary_table(events, run_id)
    make_lr_plot(events, run_id)
    make_cpu_usage_plots(events, run_id)
    make_gpu_util_plots(events, run_id)


class RunConfig():
    def __init__(self,config):
        self.embed_dim = config.get('embed-dim') if config.get('embed-dim') else 320
        self.d_hid = config.get('d-hid') if config.get('d-hid') else 1280
        self.n_head = config.get('n-head') if config.get('n-head') else 20
        self.n_layers = config.get('n-layers') if config.get('n-layers') else 6
        self.dropout = config.get('dropout') if config.get('dropout') else 0.1
        self.batch_size = config.get('batch-size') if config.get('batch-size') else 128
        self.n_train_batches = config.get('n-train-batches') if config.get('n-train-batches') else 0
        self.n_eval_batches = config.get('n-eval-batches') if config.get('n-eval-batches') else 0
        self.n_epochs = config.get('n-epochs') if config.get('n-epochs') else 3
        self.lr = config.get('lr') if config.get('lr') else 4e-4
        self.log_interval = config.get('log-interval') if config.get('log-interval') else 1
        self.num_warmup_steps  = config.get('num-warmup-steps') if config.get('num-warmup-steps') else 2000
        self.tokens_per_update = config.get('tokens-per-update') if config.get('tokens-per-update') else 2000000
        self.max_length = config.get('max-length') if config.get('max-length') else 1024
        self.token_dic = token_dic_uniref
        self.padded_collate = padded_collate_uniref
    
    def __setattr__(self, name, value):
        self.__dict__[name] = value
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment script")
    parser.add_argument('--run-id', type=str, help='The run ID for the experiment')
    args = parser.parse_args()
    run_id = args.run_id

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    args = RunConfig(config)

    args.run_id = run_id

    monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
    monitor_thread.start()

    moniter_thread2 = threading.Thread(target=monitor_cpu_load, daemon=True)
    moniter_thread2.start()

    main(args)

