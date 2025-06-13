import math
import random
from collections import namedtuple, deque
from itertools import count
import re
import numpy as np
import time
from statistics import NormalDist
import gc
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import argparse
import os
import glob
import torch
import random
from transformers.models.bert.modeling_bert import BertAttention

from models.vae import VqVaeModule
from models.seq2seq import Seq2SeqModule
from models.regression import Regression
from datasets import MidiDataset
from utils import medley_iterator
from input_representation import remi2midi
from vocab import DescriptionVocab
import numpy as np
from evaluate import meta_stats
import input_representation as ir

import pytorch_lightning as pl
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
from datasets import MidiDataModule
from vocab import RemiVocab, DescriptionVocab
from constants import (
  PAD_TOKEN, 
  EOS_TOKEN, 
  BAR_KEY, 
  POSITION_KEY,
  INSTRUMENT_KEY,
  PITCH_KEY,
  VELOCITY_KEY,
  DURATION_KEY,
  TIME_SIGNATURE_KEY,
  TEMPO_KEY,
  CHORD_KEY,
  DEFAULT_POS_PER_QUARTER,
  DEFAULT_VELOCITY_BINS,
  DEFAULT_DURATION_BINS,
  DEFAULT_TEMPO_BINS,
  DEFAULT_NOTE_DENSITY_BINS,
  DEFAULT_MEAN_VELOCITY_BINS,
  DEFAULT_MEAN_PITCH_BINS,
  DEFAULT_MEAN_DURATION_BINS,
  DEFAULT_RESOLUTION,
  NOTE_DENSITY_KEY,
  MEAN_PITCH_KEY,
  MEAN_VELOCITY_KEY,
  MEAN_DURATION_KEY
) 

def parse_args():
  parser = argparse.ArgumentParser()
  # parser.add_argument('--model', type=str, required=True, help="Model name (one of 'figaro', 'figaro-expert', 'figaro-learned', 'figaro-no-inst', 'figaro-no-chord', 'figaro-no-meta')")
  # parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint")
  parser.add_argument('--model', type=str, default="figaro-expert")
  parser.add_argument('--checkpoint', type=str, default="/data3/typer117/checkpoints/figaro-expert.ckpt")
  parser.add_argument('--checkpoints_save', type=str, default="/data3/typer117/checkpoints/checkpoints_rl")
  parser.add_argument('--vae_checkpoint', type=str, default=None, help="Path to the VQ-VAE model checkpoint (optional)")
  parser.add_argument('--lmd_dir', type=str, default='/data3/typer117/lmd_full', help="Path to the root directory of the LakhMIDI dataset")
  parser.add_argument('--max_iter', type=int, default=100000000)
  parser.add_argument('--songs_done', type=int, default=0)
  parser.add_argument('--max_bars', type=int, default=8)
  parser.add_argument('--batch_size_ep', type=int, default=8)
  parser.add_argument('--batch_size_opt', type=int, default=8)
  parser.add_argument('--verbose', type=int, default=1)
  parser.add_argument('--output_dir', type=str, default='./samples', help="Path to the output directory")
  parser.add_argument('--max_n_files', type=int, default=-1)
  parser.add_argument('--lr', type=float, default=3e-7)
  parser.add_argument('--warm_up', type=int, default=0)
  parser.add_argument('--value_lr', type=float, default=1e-5)
  parser.add_argument('--temp', type=float, default=0.8)
  parser.add_argument('--p_sampling', type=float, default=0.9)
  parser.add_argument('--p_maintainer', type=float, default=0.995)
  parser.add_argument('--alpha', type=float, default=0.99)
  parser.add_argument('--consistency', type=str, default='L')
  parser.add_argument('--baseline', type=str, default='V')
  parser.add_argument('--exploration', type=str, default='T')
  args = parser.parse_args()
  return args

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def print_memory_summary():
    print("=" * 40)
    print("🔥 GPU Memory Summary 🔥")
    print("=" * 40)
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    print("=" * 40)

def load_old_or_new_checkpoint(model_class, checkpoint):
  # assuming transformers>=4.36.0
  pl_ckpt = torch.load(checkpoint, map_location="cpu")

  kwargs = pl_ckpt['hyper_parameters']
  if 'flavor' in kwargs:
    del kwargs['flavor']
  if 'vae_run' in kwargs:
    del kwargs['vae_run']
    
  model = model_class(**kwargs)
  state_dict = pl_ckpt['state_dict']
  # position_ids are no longer saved in the state_dict starting with transformers==4.31.0
  state_dict = {k: v for k, v in state_dict.items() if not k.endswith('embeddings.position_ids') and not k.startswith('regressor')}
  try:
    # succeeds for checkpoints trained with transformers>4.13.0
    model.load_state_dict(state_dict)
  except RuntimeError:
    # work around a breaking change introduced in transformers==4.13.0, which fixed the position_embedding_type of cross-attention modules "absolute"
    config = model.transformer.decoder.bert.config
    for layer in model.transformer.decoder.bert.encoder.layer:
      layer.crossattention = BertAttention(config, position_embedding_type=config.position_embedding_type)
    model.load_state_dict(state_dict, strict=False)
  # model.freeze()
  # model.eval()
  return model


def load_model(checkpoint, model_class, device='auto'):
  if device == 'auto':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = load_old_or_new_checkpoint(model_class, checkpoint)
  model.to(device)

  return model

def save_checkpoint(model, optimizer, num_step, args):
    checkpoint = {
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, f'{args.checkpoints_save}/N{args.exploration}{args.baseline}{num_step}steps.pth')
    print(f"***Saved current model to {args.checkpoints_save}/{num_step}steps.pth***")

def load_checkpoint(num_songs, args):
    kwargs = {'d_model': 512, 'd_latent': 512, 'n_codes': 512, 'n_groups': 8, 'context_size': 256, 'lr': 0.0001, 'lr_schedule': 'sqrt_decay', 'warmup_steps': 4000, 'max_steps': 64000, 'encoder_layers': 4, 'decoder_layers': 6, 'intermediate_size': 2048, 'num_attention_heads': 8, 'description_flavor': 'description', 'description_options': None, 'use_pretrained_latent_embeddings': True}
    policy_net = Seq2SeqModule(**kwargs)

    checkpoint = torch.load(f'{args.checkpoints_save}/{num_songs}songs.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith('embeddings.position_ids')}
    config = policy_net.transformer.decoder.bert.config
    for layer in policy_net.transformer.decoder.bert.encoder.layer:
      layer.crossattention = BertAttention(config, position_embedding_type=config.position_embedding_type)
    policy_net.load_state_dict(state_dict)
    policy_net.to('cuda')
    # policy_net.to(rank)
    # policy_net = DDP(policy_net, device_ids=[rank])
    optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr, amsgrad=True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded: Resuming from song no. {num_songs}")
    return policy_net, optimizer

# Initialize process
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Cleanup after training
def cleanup():
    dist.destroy_process_group()

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward'))

class SeqCollator:
  def __init__(self, pad_token=0, context_size=512):
    self.pad_token = pad_token
    self.context_size = context_size

  def __call__(self, features):
    batch = {}

    xs = [feature['input_ids'] for feature in features]
    xs = pad_sequence(xs, batch_first=True, padding_value=self.pad_token)

    if self.context_size > 0:
      max_len = self.context_size
      max_desc_len = self.context_size
    else:
      max_len = xs.size(1)
      max_desc_len = int(1e4)

    tmp = xs[:, :(max_len + 1)]
    # labels = xs[:, :(max_len + 1)][:, 1:].clone().detach()
    xs = tmp

    seq_len = xs.size(1)
    
    batch['input_ids'] = xs
    # batch['labels'] = labels

    if 'position_ids' in features[0]:
      position_ids = [feature['position_ids'] for feature in features]
      position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0)
      batch['position_ids'] = position_ids[:, :seq_len]

    if 'bar_ids' in features[0]:
      bar_ids = [feature['bar_ids'] for feature in features]
      bar_ids = pad_sequence(bar_ids, batch_first=True, padding_value=0)
      batch['bar_ids'] = bar_ids[:, :seq_len]

    if 'current_id' in features[0]:
      current_ids = [feature['current_id'].unsqueeze(0) for feature in features]
      batch['current_ids'] = torch.cat(current_ids)

    if 'latents' in features[0]:
      latents = [feature['latents'] for feature in features]
      latents = pad_sequence(latents, batch_first=True, padding_value=0.0)
      batch['latents'] = latents[:, :max_desc_len]
    
    if 'codes' in features[0]:
      codes = [feature['codes'] for feature in features]
      codes = pad_sequence(codes, batch_first=True, padding_value=0)
      batch['codes'] = codes[:, :max_desc_len]

    if 'description' in features[0]:
      description = [feature['description'] for feature in features]
      description = pad_sequence(description, batch_first=True, padding_value=self.pad_token)
      desc = description[:, :max_desc_len]
      batch['description'] = desc

      if 'desc_bar_ids' in features[0]:
        desc_len = desc.size(1)
        desc_bar_ids = [feature['desc_bar_ids'] for feature in features]
        desc_bar_ids = pad_sequence(desc_bar_ids, batch_first=True, padding_value=0)
        batch['desc_bar_ids'] = desc_bar_ids[:, :desc_len]

    if 'file' in features[0]:
      batch['files'] = [feature['file'] for feature in features]
    
    # print(f"Batch received by collate_fn: {batch}")
    return batch

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = []

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    # def sample(self, batch_size):
    #     return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def top_p_sampling(logits, args, temp, filter_value=-float('Inf')):
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > args.p_sampling
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value

    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))/temp

    pred_token = torch.multinomial(F.softmax(logits, -1), 1).view(-1) # [BATCH_SIZE]

    return pred_token

def scheduled_temperature(iteration, total_iterations=20000, start_temp=2.0, end_temp=0.8):
    """Linearly schedules temperature from start_temp to end_temp over total_iterations."""
    if iteration <= total_iterations:
        return start_temp + (iteration / total_iterations) * (end_temp - start_temp)
    else:
        return end_temp  

def scheduled_epsilon(iteration, total_iterations=20000, start_temp=0.2, end_temp=0.0):
    """Linearly schedules temperature from start_temp to end_temp over total_iterations."""
    if iteration <= total_iterations:
        return start_temp + (iteration / total_iterations) * (end_temp - start_temp)
    else:
        return end_temp  

def select_action_(rank, state, model, pretrained_model, current_ids, num_bars, step_idx, args):
  """_summary_

  Args:
      state (dict{'input_ids', 'bar_ids', 'position_ids', 'description', 'desc_bar_ids'}): 고정되지 않은 사이즈

  Returns:
      next_token_ids (tensor(int)): (batch_size, 1)
      next_tokens (tensor(str)): (batch_size, 1)
  """
  x = state['input_ids'].clone()
  bar_ids = state['bar_ids'].clone()
  position_ids = state['position_ids'].clone()
  z = state['description'].clone()
  desc_bar_ids = state['desc_bar_ids'].clone()
  next_token_ids = torch.zeros(args.batch_size_ep)
  maintainer = torch.zeros(args.batch_size_ep, dtype=torch.float32)

  idx = current_ids.reshape(args.batch_size_ep,1,1).expand(args.batch_size_ep,1,1357) 
  
  encoder_hidden_states_ = pretrained_model.encode(z, desc_bar_ids)
  model_output_ = pretrained_model.decode(x, bar_ids=bar_ids, position_ids=position_ids, encoder_hidden_states=encoder_hidden_states_)
  logits_ = model_output_.gather(1, idx).squeeze(1)
  prob = F.softmax(logits_, dim=-1)

  sorted_logits, sorted_indices = torch.sort(logits_, descending=True)
  cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
  sorted_indices_to_remove = cumulative_probs > args.p_maintainer
  sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
  sorted_indices_to_remove[..., 0] = 0  

  if args.exploration == 'E':
    epsilon = scheduled_epsilon(num_bars)
    if random.random() < epsilon:
      for j in range(args.batch_size_ep):
        action_idx = torch.randint(0, sorted_indices[j][~sorted_indices_to_remove[j]].shape[0], (1,)).item()
        next_token_ids[j] = sorted_indices[j][~sorted_indices_to_remove[j]][action_idx]  
      next_token_ids = next_token_ids.to('cpu')
      if args.consistency == 'L':
        maintainer[j] = prob[j][next_token_ids[j].item()]

    else:
      with torch.no_grad():
          encoder_hidden_states = model.module.encode(z, desc_bar_ids)
          logits = model.module.decode(x, bar_ids=bar_ids, position_ids=position_ids, encoder_hidden_states=encoder_hidden_states)
          logits = logits.gather(1, idx).squeeze(1) / args.temp
          pr = F.softmax(logits, dim=-1)
          pr = pr.view(-1, pr.size(-1))
          next_token_ids = torch.multinomial(pr, 1).view(-1).to('cpu')

          for j in range(args.batch_size_ep):
            if args.consistency == 'N':
              if next_token_ids[j] not in sorted_indices[j][~sorted_indices_to_remove[j]]:
                  maintainer[j] = -1
            elif args.consistency == 'L':
              maintainer[j] = prob[j][next_token_ids[j].item()]


  elif args.exploration == 'T':
    with torch.no_grad():
      encoder_hidden_states = model.module.encode(z, desc_bar_ids)
      logits = model.module.decode(x, bar_ids=bar_ids, position_ids=position_ids, encoder_hidden_states=encoder_hidden_states)
      logits = logits.gather(1, idx).squeeze(1)
      temperature = scheduled_temperature(num_bars)
      next_token_ids = top_p_sampling(logits.clone(), args, temperature).to('cpu')
      for j in range(args.batch_size_ep):
        if args.consistency == 'N':
          if next_token_ids[j] not in sorted_indices[j][~sorted_indices_to_remove[j]]:
              maintainer[j] = -1
        elif args.consistency == 'L':
          maintainer[j] = prob[j][next_token_ids[j].item()]

    # if step_idx % 50 == 0:
    #   dist = torch.distributions.Categorical(probs=F.softmax(logits,-1))
    #   entropy = torch.mean(dist.entropy())
    #   with open("entropy.txt", "a") as file:
    #     file.write(str(entropy.item()))
    #     file.write("\n")
  next_tokens = model.module.vocab.decode(next_token_ids)
  return next_token_ids.to(rank), next_tokens, maintainer
  
def select_action(rank, state, model, current_ids, args, step_idx):
  """_summary_

  Args:
      state (dict{'input_ids', 'bar_ids', 'position_ids', 'description', 'desc_bar_ids'}): 고정되지 않은 사이즈

  Returns:
      next_token_ids (tensor(int)): (batch_size, 1)
      next_tokens (tensor(str)): (batch_size, 1)
  """
  x = state['input_ids'].clone()
  bar_ids = state['bar_ids'].clone()
  position_ids = state['position_ids'].clone()
  z = state['description'].clone()
  desc_bar_ids = state['desc_bar_ids'].clone()

  idx = current_ids.reshape(args.batch_size_ep,1,1).expand(args.batch_size_ep,1,1357) 
  
  with torch.no_grad():
    encoder_hidden_states = model.module.encode(z, desc_bar_ids)
    logits = model.module.decode(x, bar_ids=bar_ids, position_ids=position_ids, encoder_hidden_states=encoder_hidden_states)
    logits = logits.gather(1, idx).squeeze(1) / args.temp
    pr = F.softmax(logits, dim=-1)
    pr = pr.view(-1, pr.size(-1))
    
    if step_idx % 50 == 0:
      dist = torch.distributions.Categorical(probs=pr)
      entropy = torch.mean(dist.entropy())
      with open("entropy.txt", "a") as file:
        file.write(str(entropy.item()))
        file.write("\n")
        
    next_token_ids = torch.multinomial(pr, 1).view(-1).to('cpu')

    next_tokens = model.module.vocab.decode(next_token_ids)
    return next_token_ids.to(rank), next_tokens

def overlapping_area(mu1, sigma1, mu2, sigma2, eps=0.01):
  sigma1, sigma2 = max(eps, sigma1), max(eps, sigma2)
  return NormalDist(mu=mu1, sigma=sigma1).overlap(NormalDist(mu=mu2, sigma=sigma2))  
  
def step(rank,
  state, 
  action, 
  z, 
  desc_bar_ids,
  model,
  vocab, 
  terminated,
  events_full,
  bar_ids_full,
  current_ids,
  num_bars,
  meta_stats,
  baseline,
  maintainer,
  args,
  pad_token=PAD_TOKEN, 
  eos_token=EOS_TOKEN
):
  """_summary_

  Args:
      state (dict{'input_ids', 'bar_ids', 'position_ids', 'description', 'desc_bar_ids'}): 고정되지 않은 사이즈
      action (next_token_ids, next_tokens): 
      z (_type_): true description
      desc_bar_ids (_type_): true description
      terminated (tensor): (batch_size)
      pad_token (_type_, optional): _description_. Defaults to PAD_TOKEN.
      eos_token (_type_, optional): _description_. Defaults to EOS_TOKEN.
  """
  pad_token_id = model.module.vocab.to_i(pad_token)
  eos_token_id = model.module.vocab.to_i(eos_token)
  x = state['input_ids'].clone()
  bar_ids = state['bar_ids'].clone()
  position_ids = state['position_ids'].clone()
  z_ = state['description'].clone()
  desc_bar_ids_ = state['desc_bar_ids'].clone()
  next_token_ids, next_tokens = action[0].clone(), action[1]
  index = current_ids.reshape(args.batch_size_ep,1)
  batch_size = x.shape[0]
  is_done = terminated
  reward_rule = maintainer
  reward_metric = torch.zeros(batch_size, dtype=torch.float32)
  bar_done = torch.zeros(args.batch_size_ep, dtype=torch.bool)
  context_size = model.module.context_size
  curr_bars = bar_ids[:, 0]
  next_bars = torch.tensor([1 if f'{BAR_KEY}_' in token else 0 for token in next_tokens], dtype=torch.int).to(rank)
  next_bar_ids = bar_ids.gather(1, index).clone().squeeze(1) + next_bars
  
  next_positions = [f"{POSITION_KEY}_0" if f'{BAR_KEY}_' in token else token for token in next_tokens]
  next_positions = [int(token.split('_')[-1]) if f'{POSITION_KEY}_' in token else None for token in next_positions]
  next_positions = [pos if next_pos is None else next_pos for pos, next_pos in zip(position_ids.gather(1, index).squeeze(1), next_positions)]
  next_position_ids = torch.tensor(next_positions, dtype=torch.int).to(rank)

  is_done.masked_fill_(next_token_ids == eos_token_id, True)
  is_done.masked_fill_(next_bar_ids >= args.max_bars + 1, True)
    
  for i,j in zip(range(args.batch_size_ep), current_ids):
    if j == model.module.context_size-1:
      x[i] = torch.cat([x[i], next_token_ids[i].clone().unsqueeze(0)])[-context_size:]
      bar_ids[i] = torch.cat([bar_ids[i], next_bar_ids[i].clone().unsqueeze(0)])[-context_size:]
      position_ids[i] = torch.cat([position_ids[i], next_position_ids[i].unsqueeze(0)])[-context_size:]

      if (position_ids[i,-1] < position_ids[i,-2]) & (f'{BAR_KEY}_' not in action[1][i]):
        reward_rule[i] += -1.0
    else:
      x[i,j+1] = next_token_ids[i].clone()
      bar_ids[i,j+1] = next_bar_ids[i].clone()
      position_ids[i,j+1] = next_position_ids[i]

      if (position_ids[i,j+1] < position_ids[i,j]) & (f'{BAR_KEY}_' not in action[1][i]):
        reward_rule[i] += -1.0

  # z, desc_bar_ids scrolling
  next_bars = bar_ids[:, 0]
  bars_changed = not (next_bars == curr_bars).all()
  
  if bars_changed:
    z_ = torch.zeros(batch_size, context_size, dtype=torch.int).to(rank)
    desc_bar_ids_ = torch.zeros(batch_size, context_size, dtype=torch.int).to(rank)

    for j in range(batch_size):
      curr_bar = bar_ids[j, 0]
      indices = torch.nonzero(desc_bar_ids[j] == curr_bar)
      if indices.size(0) > 0:
        idx = indices[0, 0]
      else:
        idx = z.size(1) - 1

      offset = min(context_size, z.size(1) - idx)

      z_[j, :offset] = z[j, idx:idx+offset]
      desc_bar_ids_[j, :offset] = desc_bar_ids[j, idx:idx+offset].clone()

    z_, desc_bar_ids_ = z_.to(rank), desc_bar_ids_.to(rank)

  for i, j in zip(range(batch_size), current_ids):
    if f'{BAR_KEY}_' in action[1][i] and int(action[1][i].split('_')[-1]) != 1:
      bar_done[i] = True
      if j == model.module.context_size-1:
        current_bar = bar_ids[i][j-1]
      else: 
        current_bar = bar_ids[i][j]
      
      events_indices = torch.nonzero(bar_ids_full[i] == current_bar).to('cpu')
      # desc_indices = torch.nonzero(desc_bar_ids[i] == current_bar).to('cpu')
      curr_bar_seq = events_full[i].to('cpu')[events_indices].clone()

      # curr_bar_desc = vocab.decode(z[i].to('cpu')[desc_indices])
      curr_bar_seq = [model.module.vocab.decode(x)[0] for x in curr_bar_seq]
      
      if num_bars % 5 == 1:
        with open("bar_seq.txt", 'w') as file:
          for token in curr_bar_seq:
            file.write(token)
            file.write("\n")

      nd, pm, pstd, vm, vstd, dm, dstd, ts, inst, ch = get_meta(curr_bar_seq)
      try:
        nd_org = meta_stats[i]['note_density'][current_bar-1]
        pm_org = meta_stats[i]['pitch_mean'][current_bar-1]
        pstd_org = meta_stats[i]['pitch_std'][current_bar-1]
        vm_org = meta_stats[i]['velocity_mean'][current_bar-1]
        vstd_org = meta_stats[i]['velocity_std'][current_bar-1]
        dm_org = meta_stats[i]['duration_mean'][current_bar-1]
        dstd_org = meta_stats[i]['duration_std'][current_bar-1]
        ts_org = meta_stats[i]['time_signature'][current_bar-1]
        inst_org = meta_stats[i]['instruments'][current_bar-1]
        ch_org = meta_stats[i]['chords'][current_bar-1]

        if len(ts) == 0 and len(ts_org) == 0:
          reward_ts = baseline['ts']
        elif len(ts) > 0 and len(ts_org) > 0:
          reward_ts = float(ts[0] == ts_org[0])
        else:
          reward_ts = 0
        reward_if1 = compute_f1(inst,inst_org)
        reward_cf1 = compute_f1(ch, ch_org)

        reward_nd = 1/((np.abs(nd - nd_org) / np.mean(meta_stats[i]['note_density']))+1)
        reward_poa = overlapping_area(pm, pstd, pm_org, pstd_org)
        if np.isnan(reward_poa):
          reward_poa = baseline['poa']
          # if np.isnan(pm) and np.isnan(pm_org):
          #   reward_poa = 1.0
          # else:
          #   reward_poa = baseline['poa'] - 0.23
        reward_voa = overlapping_area(vm, vstd, vm_org, vstd_org)
        if np.isnan(reward_voa):
          reward_voa = baseline['voa']
        reward_doa = overlapping_area(dm, dstd, dm_org, dstd_org)
        if np.isnan(reward_doa):
          reward_doa = baseline['doa']        

      except:
        reward_nd = baseline['nd']
        reward_poa = baseline['poa']
        reward_voa = baseline['voa']
        reward_doa = baseline['doa']
        reward_ts = baseline['ts']
        reward_if1 = baseline['if1']
        reward_cf1 = baseline['cf1']

      # with open("reward_poa.txt", 'a') as file:
      #   file.write(str(reward_poa))
      #   file.write("\n")        
      # with open("reward_nd.txt", 'a') as file:
      #   file.write(str(reward_nd))
      #   file.write("\n")
      # with open("reward_voa.txt", 'a') as file:
      #   file.write(str(reward_voa))
      #   file.write("\n")        
      # with open("reward_doa.txt", 'a') as file:
      #   file.write(str(reward_doa))
      #   file.write("\n")
      # with open("reward_ts.txt", 'a') as file:
      #   file.write(str(reward_ts))
      #   file.write("\n")
      # with open("reward_if1.txt", 'a') as file:
      #   file.write(str(reward_if1))
      #   file.write("\n")        
      # with open("reward_cf1.txt", 'a') as file:
      #   file.write(str(reward_cf1))
      #   file.write("\n")
        
      reward_metric[i] = reward_nd + reward_poa + reward_voa + reward_doa + reward_ts + reward_if1 + reward_cf1
      # reward_metric[i] = reward_nd * reward_poa * reward_voa * reward_doa * 10

      baseline['nd'] = args.alpha * baseline['nd'] + (1 - args.alpha) * reward_nd
      baseline['poa'] = args.alpha * baseline['poa'] + (1 - args.alpha) * reward_poa
      baseline['voa'] = args.alpha * baseline['voa'] + (1 - args.alpha) * reward_voa
      baseline['doa'] = args.alpha * baseline['doa'] + (1 - args.alpha) * reward_doa
      baseline['ts'] = args.alpha * baseline['ts'] + (1 - args.alpha) * reward_ts
      baseline['if1'] = args.alpha * baseline['if1'] + (1 - args.alpha) * reward_if1
      baseline['cf1'] = args.alpha * baseline['cf1'] + (1 - args.alpha) * reward_cf1
      
      if int(action[1][i].split('_')[-1]) != next_bar_ids[i]:
        reward_rule[i] += -1.0
      
  reward_rule = reward_rule.to(rank)
  reward_metric = reward_metric.to(rank)
      
  return {'input_ids' : x,
          'bar_ids' : bar_ids,
          'position_ids' : position_ids,
          'description' : z_,
          'desc_bar_ids' : desc_bar_ids_}, reward_rule, reward_metric, is_done, bar_done, baseline

def compute_f1(set1,set2):
  tp = len(set1 & set2)
  try:
    p = tp/len(set1)
    r = tp/len(set2)
  except:
    return 0
  if p+r > 0:
    f1 = 2*p*r / (p+r)
  else:
    f1 = 0
  return f1

def get_positions_per_bar(bar_seq):
  time_sig = [token.split("_")[-1] for token in bar_seq if token.startswith(f'{TIME_SIGNATURE_KEY}_')]
  if len(time_sig) == 0:
    return None
  time_sig = time_sig[0]
  numerator, denominator = [int(x) for x in time_sig.split("/")]
  quarters_per_bar = 4 * numerator / denominator
  positions_per_bar = int(DEFAULT_POS_PER_QUARTER * quarters_per_bar)
  return positions_per_bar

def get_meta(bar_seq):
  ppb = get_positions_per_bar(bar_seq)
  notes = []
  for i in range(len(bar_seq) - 4):
    notes.append(bar_seq[i:i+5])
  notes = [note for note in notes if (f'{POSITION_KEY}_'in note[0] and 
                                      f'{INSTRUMENT_KEY}_'in note[1] and 
                                      f'{PITCH_KEY}_'in note[2] and 
                                      f'{VELOCITY_KEY}_'in note[3] and 
                                      f'{DURATION_KEY}_'in note[4])]

  if (len(notes) > 0) & (ppb != None):
    note_density = len(notes)/(ppb/DEFAULT_POS_PER_QUARTER)
  else:
    note_density = 100

  notes_ = [note for note in notes if note[1].split('_')[-1] != 'drum']
  pitches = [int(note[2].split('_')[-1]) for note in notes_]
  pitch_mean = np.mean(pitches) if len(pitches) else np.nan
  pitch_std = np.std(pitches) if len(pitches) else np.nan

  velocities = [min(127,DEFAULT_VELOCITY_BINS[int(note[3].split('_')[-1])]) for note in notes_]
  velocity_mean = np.mean(velocities) if len(velocities) else np.nan
  velocity_std = np.std(velocities) if len(velocities) else np.nan

  durations = [DEFAULT_DURATION_BINS[int(note[4].split('_')[-1])] for note in notes_]
  duration_mean = np.mean(durations) if len(durations) else np.nan
  duration_std = np.std(durations) if len(durations) else np.nan

  ts = [token for token in bar_seq if f'{TIME_SIGNATURE_KEY}_' in token]
  inst = set([token for token in bar_seq if f'{INSTRUMENT_KEY}_' in token])
  chord = set([token for token in bar_seq if f'{CHORD_KEY}_' in token])
  
  return note_density, pitch_mean, pitch_std, velocity_mean, velocity_std, duration_mean, duration_std, ts, inst, chord
  
def dataloader_setup(rank, world_size, model, coll, args):
  # midi_files = glob.glob(os.path.join(args.lmd_dir, '**/*.mid'), recursive=True)
  # dm = model.get_datamodule(midi_files)
  # dm.setup('train')
  midi_files = glob.glob(os.path.join(args.lmd_dir, '**/*.mid'), recursive=True)
  dm = model.get_datamodule(midi_files)
  dm.setup('train')
  midi_files = dm.train_ds.files

  if rank == 0:
    print(f"Total size of training set: {len(midi_files)}")
    # for i in midi_files[:10]:
    #   print(i)
  random.seed(42)
  random.shuffle(midi_files)
  if args.max_n_files > 0:
    midi_files = midi_files[:args.max_n_files]  

  total_len = len(midi_files)
  shard_size = total_len // world_size
  start_idx = rank * shard_size
  end_idx = min(start_idx + shard_size, total_len)
  midi_files = midi_files[start_idx:end_idx]
  midi_files = midi_files[args.songs_done:] + midi_files[:args.songs_done]
  print(f"Trainig set size for rank_{rank}: {len(midi_files)}")

  dataset = MidiDataset(
    midi_files,
    max_len=-1,
    description_flavor=model.description_flavor,
    max_bars=model.context_size,
  )
  dl = DataLoader(dataset, batch_size=1, collate_fn=coll)
  
  return dl
  
def optimize_model(rank, model, value_net, episode_idx, reward, memory, optimizer, value_optimizer, scaler, num_bars, baseline, args):
  criterion = nn.MSELoss()
  idx = episode_idx
  try:
    episode_memory = memory[idx]
    r = reward
    
    for i in range(0, len(episode_memory), args.batch_size_opt):
      transitions = episode_memory.memory[i:i+args.batch_size_opt]
      batch_size = len(transitions) 
    
      batch = Transition(*zip(*transitions))
      x = torch.cat([state['input_ids'].unsqueeze(0) for state in batch.state], dim=0).clone()
      bar_ids = torch.cat([state['bar_ids'].unsqueeze(0) for state in batch.state], dim=0).clone()
      position_ids = torch.cat([state['position_ids'].unsqueeze(0) for state in batch.state], dim=0).clone()
      z = torch.cat([state['description'].unsqueeze(0) for state in batch.state], dim=0).clone()
      desc_bar_ids = torch.cat([state['desc_bar_ids'].unsqueeze(0) for state in batch.state], dim=0).clone()
      sequence_index = torch.cat([state['current_id'].unsqueeze(0) for state in batch.state]).clone()
      
      action_batch = torch.cat(batch.action).clone()
      reward_rule_batch = torch.cat(batch.reward).clone().unsqueeze(-1)
  
      sequence_index = sequence_index.reshape(batch_size,1,1).expand(batch_size,1,1357)  
      action_index = action_batch.reshape(batch_size,1).long()

      if args.baseline == "V":
        encoder_hidden_states_v = value_net.module.encode(z, desc_bar_ids)
        model_output_v = value_net.module.decode(x, bar_ids=bar_ids, position_ids=position_ids, encoder_hidden_states=encoder_hidden_states_v)
        loss_v = criterion(model_output_v, r.repeat(len(model_output_v)))

        # with open("lossv.txt", 'a') as file:
        #   file.write(str(loss_v.item()))
        #   file.write("\n")
        loss_v.backward()
        torch.nn.utils.clip_grad_value_(value_net.parameters(), 1.0)
        value_optimizer.step()
        value_optimizer.zero_grad()

        if num_bars < args.warm_up:
          continue

        base = model_output_v.unsqueeze(-1).detach()
      
      elif args.baseline == 'M':
        if num_bars < args.warm_up:
          continue
        base = baseline['nd'] + baseline['poa'] + baseline['voa'] + baseline['doa'] + baseline['ts'] + baseline['if1'] + baseline['cf1']

      with torch.cuda.amp.autocast():
        encoder_hidden_states = model.module.encode(z, desc_bar_ids)
        model_output = model.module.decode(x, bar_ids=bar_ids, position_ids=position_ids, encoder_hidden_states=encoder_hidden_states)
        logits = model_output.gather(1, sequence_index).squeeze(1)
        prob = F.softmax(logits, dim=-1)
        pr = torch.log(prob.gather(1,action_index))

        #------------------------------------------------------------------
        loss = (- (r + reward_rule_batch - base) * pr).sum()
      scaler.scale(loss).backward()

    if num_bars >= args.warm_up:
      torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
      
      scaler.step(optimizer)
      scaler.update()    

      optimizer.zero_grad()
    #-------------------------------------------------------
  except torch.cuda.OutOfMemoryError:
    print(f"OOM error on bar no.{num_bars}, skipping gradient step for current bar...")

  memory[idx].memory = []


def train(rank, world_size):
  start_time = time.time()
  args = parse_args()
  if rank == 0:
    print(args)
  setup(rank, world_size)
  print(f"Rank {rank}: Begin training")
  
  # policy_net, optimizer = load_checkpoint(args.songs_done, args)
  policy_net = load_model(args.checkpoint, Seq2SeqModule)

  policy_net = policy_net.to(rank)
  policy_net = DDP(policy_net, device_ids=[rank])

  optimizer = optim.AdamW(policy_net.parameters(), lr=args.lr, amsgrad=True)
  optimizer.zero_grad()

  value_net = load_model(args.checkpoint, Regression)
  value_net = value_net.to(rank)
  value_net = DDP(value_net, device_ids=[rank])

  value_optimizer = optim.AdamW(value_net.parameters(), lr=args.value_lr, amsgrad=True)
  value_optimizer.zero_grad()

  scaler = torch.cuda.amp.GradScaler()

  fixed_net = load_model(args.checkpoint, Seq2SeqModule)
  fixed_net.freeze()
  fixed_net = fixed_net.to(rank)

  memory = []
  baseline = {'nd':0.80, 'poa':0.71, 'voa': 0.61, 'doa': 0.47, 'ts': 0.99, 'if1': 0.95, 'cf1': 0.6}
  baseline_keys = ['nd','poa','voa','doa']
  
  for i in range(args.batch_size_ep):
    memory.append(ReplayMemory(10000))  
  vocab = DescriptionVocab()
  coll = SeqCollator(context_size=-1)
  
  dl = dataloader_setup(rank, world_size, policy_net.module, coll, args)
  dl_iter = iter(dl)
  initial_context = 1
  epoch = 1
  num_songs = args.songs_done
  
  batch = []
  for i in range(args.batch_size_ep):
    try:
      data = next(dl_iter)
      num_songs += 1
    except StopIteration:
      epoch += 1
      dl_iter = iter(dl)
      data = next(dl_iter)
      num_songs = 1
    batch.append({key:data[key].squeeze(0) for key in ['input_ids', 'bar_ids', 'position_ids', 'description', 'desc_bar_ids']})
    # if rank == 0:
    #   print(data['files'])

  batch = coll(batch)
  meta_stats = []
  xs = batch['input_ids'].detach().cpu()
  sequences = [policy_net.module.vocab.decode(x) for x in xs]
  meta_key = [
    'note_density', 
    'pitch_mean', 
    'pitch_std', 
    'velocity_mean', 
    'velocity_std', 
    'duration_mean', 
    'duration_std',
    'time_signature',
    'instruments',
    'chords']
  # batch에 대한 for-loop
  for sequence_ in sequences:
    meta_per_song = {key:[] for key in meta_key}
    sequence = sequence_[1:]
    bars = [1 if 'Bar_' in token else 0 for token in sequence]
    bar_ids_meta = np.cumsum(bars) - 1
    bar_seqs = [[] for _ in range(bar_ids_meta[-1] + 1)]
    for i, token in enumerate(sequence):
      bar_seqs[bar_ids_meta[i]].append(token)
    bar_seqs = bar_seqs[:args.max_bars]
    
    # bar에 대한 for-loop
    for bar_seq in bar_seqs:
      note_density, pitch_mean, pitch_std, velocity_mean, velocity_std, duration_mean, duration_std, time_signature, instruments, chords = get_meta(bar_seq)
      meta_per_song['note_density'].append(note_density)
      meta_per_song['pitch_mean'].append(pitch_mean)
      meta_per_song['pitch_std'].append(pitch_std)
      meta_per_song['velocity_mean'].append(velocity_mean)
      meta_per_song['velocity_std'].append(velocity_std)
      meta_per_song['duration_mean'].append(duration_mean)
      meta_per_song['duration_std'].append(duration_std)
      meta_per_song['time_signature'].append(time_signature)
      meta_per_song['instruments'].append(instruments)
      meta_per_song['chords'].append(chords)
    meta_stats.append(meta_per_song)   

  print(f"Rank {rank}: Dataloader ready")
  
  batch_ = { key: torch.cat([batch[key][:, :initial_context],torch.zeros(args.batch_size_ep,policy_net.module.context_size-1, dtype=torch.int)],dim=1).to(rank) for key in ['input_ids', 'bar_ids', 'position_ids'] }
  batch_.update({ key: batch[key][:, :policy_net.module.context_size].to(rank) for key in ['description', 'desc_bar_ids'] })
  z = batch['description'].to(rank)
  desc_bar_ids = batch['desc_bar_ids'].to(rank)
  state = batch_
  terminated = torch.zeros(args.batch_size_ep, dtype=torch.bool).to(rank)
  events = batch_['input_ids'][:, :initial_context].clone().to(rank)
  bar_ids = batch_['bar_ids'][:, :initial_context].clone().to(rank)
  
  batch_size = args.batch_size_ep
  current_steps = torch.zeros(args.batch_size_ep, dtype=torch.int64).to(rank)
  current_ids = torch.zeros(args.batch_size_ep, dtype=torch.int64).to(rank)
  desc_len = torch.tensor([len(desc) for desc in z]).to(rank)
  bar_length = torch.zeros(args.batch_size_ep, dtype=torch.int64).to(rank)
  bar_no = torch.zeros(args.batch_size_ep, dtype=torch.int64).to(rank) + 1
  
  step_ = 0
  num_bars = 0
  print(f"Rank {rank}: Begin episode")
  if rank==0:
    print_memory_summary()

  for temp_idx in range(args.max_iter):
    
    if (rank == 0) and (step_%500==0):
      save_checkpoint(policy_net, optimizer, step_, args)
      with open("log.txt", 'a') as file:
        file.write(f"***Saved current model to {args.checkpoints_save}/{step_}steps.pth***")
        file.write("\n")
    
    if (rank == 0) and (step_%100==0):
      current_time = time.time()
      elapsed_time = current_time - start_time
      formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
      with open("log.txt", 'a') as file:
        file.write(f"Epoch:{epoch} | Songs:{num_songs} | Bars:{num_bars} | Steps:{step_} | Avg_reward:{baseline['nd']:.3f}/{baseline['poa']:.3f}/{baseline['voa']:.3f}/{baseline['doa']:.3f}/{baseline['ts']:.3f}/{baseline['if1']:.3f}/{baseline['cf1']:.3f}/{np.sum([baseline[key] for key in baseline_keys]):.3f} | Ex_time:{formatted_time}")
        file.write("\n")
    step_ += 1
    ######

    action = select_action_(rank, state, policy_net, fixed_net, current_ids, num_bars, step_, args)
    maintainer = action[2]
    action = action[0:2]
    for k,length in enumerate(bar_length):
      if length >= 1000:
        action[1][k] = f'{BAR_KEY}_{bar_no[k]+1}'
        action[0][k] = int(bar_no[k]) + 605
        print(action[0][k])

    next_bars = torch.tensor([1 if f'{BAR_KEY}_' in token else 0 for token in action[1]], dtype=torch.int).to(rank)
    
    idx = current_steps.reshape(args.batch_size_ep,1)
    next_bar_ids = bar_ids.gather(1, idx).clone().squeeze(1) + next_bars
    # if rank == 0:
    #   print(f"step:{step_}: ", action[1], action[0])
    events = torch.cat([events, torch.zeros(args.batch_size_ep,1).to(rank)], dim=1)
    bar_ids = torch.cat([bar_ids, torch.zeros(args.batch_size_ep,1).to(rank)], dim=1)
    for i,j in zip(range(args.batch_size_ep), current_steps):
      if j == current_steps.max():
        events[i,-1] = action[0][i].clone()
        bar_ids[i,-1] = next_bar_ids[i].clone()
      else:
        events[i,j+1] = action[0][i].clone()
        bar_ids[i,j+1] = next_bar_ids[i].clone()
    #####

    next_state, reward_rule, reward_metric, terminated, bar_done, baseline = step(rank, state, action, z, desc_bar_ids, policy_net, vocab, terminated, events, bar_ids, current_ids, num_bars, meta_stats, baseline, maintainer, args)
    bar_length = bar_length + 1
    # memory push
    for j,k in zip(range(batch_size),current_ids):
      temp_state = {key:state[key][j] for key in ['input_ids', 'bar_ids', 'position_ids', 'description', 'desc_bar_ids']}
      temp_state.update({'current_id': k})
       

      memory[j].push(temp_state, action[0][j].unsqueeze(0), reward_rule[j].unsqueeze(0)) ## Erase it for baseline comparison
      
      if bar_done[j]:
        num_bars += 1
        bar_length[j] = 1
        bar_no[j] += 1
        # with open("reward.txt", 'a') as file:
          # file.write(str(reward_metric[j].item()))
          # file.write("\n")
        #####
        optimize_model(rank, policy_net, value_net, j, reward_metric[j], memory, optimizer, value_optimizer, scaler, num_bars, baseline, args) ## Erase it for baseline comparison

    state = {key:next_state[key][~terminated] for key in ['input_ids', 'bar_ids', 'position_ids', 'description', 'desc_bar_ids']}
    events = events[~terminated]
    bar_ids = bar_ids[~terminated]
    current_ids = torch.tensor([id + 1 if id < policy_net.module.context_size-1 else id for id in current_ids[~terminated]], dtype=torch.int64).to(rank)
    current_steps = current_steps[~terminated] + 1
    desc_len = desc_len[~terminated]
    z = z[~terminated]
    desc_bar_ids = desc_bar_ids[~terminated]
    bar_length = bar_length[~terminated]
    bar_no = bar_no[~terminated]
    meta_stats = [item for item, keep in zip(meta_stats, terminated) if not keep]
    
    for _ in terminated:
      if _ == 1:
        try:
          new_data = next(dl_iter)
          num_songs += 1
        except StopIteration:
          epoch += 1
          num_songs = 1
          dl_iter = iter(dl)
          new_data = next(dl_iter)
        
        new_x = new_data['input_ids'][0].detach().cpu()
        new_sequence = policy_net.module.vocab.decode(new_x)
        meta_per_song = {key:[] for key in meta_key}
        new_sequence = new_sequence[1:]
        bars = [1 if 'Bar_' in token else 0 for token in new_sequence]
        bar_ids_meta = np.cumsum(bars) - 1
        bar_seqs = [[] for _ in range(bar_ids_meta[-1] + 1)]
        for i, token in enumerate(new_sequence):
          bar_seqs[bar_ids_meta[i]].append(token)
        bar_seqs = bar_seqs[:args.max_bars]
    
        # bar에 대한 for-loop
        for bar_seq in bar_seqs:
          note_density, pitch_mean, pitch_std, velocity_mean, velocity_std, duration_mean, duration_std, time_signature, instruments, chords = get_meta(bar_seq)
          meta_per_song['note_density'].append(note_density)
          meta_per_song['pitch_mean'].append(pitch_mean)
          meta_per_song['pitch_std'].append(pitch_std)
          meta_per_song['velocity_mean'].append(velocity_mean)
          meta_per_song['velocity_std'].append(velocity_std)
          meta_per_song['duration_mean'].append(duration_mean)
          meta_per_song['duration_std'].append(duration_std)
          meta_per_song['time_signature'].append(time_signature)
          meta_per_song['instruments'].append(instruments)
          meta_per_song['chords'].append(chords)
        meta_stats.append(meta_per_song)
        
        new_data_ = { key: torch.cat([new_data[key][:, :initial_context],torch.zeros(1, policy_net.module.context_size-1, dtype=torch.int)],dim=1).to(rank) for key in ['input_ids', 'bar_ids', 'position_ids']}
        new_desc_len = len(new_data['description'][0])
        if new_desc_len >= policy_net.module.context_size:
          new_data_.update({ key: new_data[key][:, :policy_net.module.context_size].to(rank) for key in ['description', 'desc_bar_ids'] })
        else:
          new_data_.update({ key: torch.cat([new_data[key], torch.zeros(1, policy_net.module.context_size-new_desc_len, dtype=torch.int)], dim=1).to(rank) for key in ['description', 'desc_bar_ids'] })
        new_z = new_data['description'][0].to(rank)
        new_desc_bar_ids = new_data['desc_bar_ids'][0].to(rank)
        new_z_len = len(new_z)
        new_events = new_data['input_ids'][0, :initial_context].clone()
        new_bar_ids = new_data['bar_ids'][0, :initial_context].clone()
        
        state = { key: torch.cat([state[key], new_data_[key]], dim=0) for key in ['input_ids', 'bar_ids', 'position_ids', 'description', 'desc_bar_ids']}
        
        events = [x[:y+1] for x,y in zip(events,current_steps)]
        events.append(new_events)
        events = pad_sequence(events, batch_first=True, padding_value=0)
        
        bar_ids = [x[:y+1] for x,y in zip(bar_ids,current_steps)]
        bar_ids.append(new_bar_ids)
        bar_ids = pad_sequence(bar_ids, batch_first=True, padding_value=0).to(rank)
        
        z = [x[:y] for x,y in zip(z,desc_len)]
        z.append(new_z)
        z = pad_sequence(z, batch_first=True, padding_value=0)
        
        desc_bar_ids = [x[:y] for x,y in zip(desc_bar_ids,desc_len)]
        desc_bar_ids.append(new_desc_bar_ids)
        desc_bar_ids = pad_sequence(desc_bar_ids, batch_first=True, padding_value=0)
        
        current_ids = torch.cat([current_ids, torch.zeros(1, dtype=torch.int64).to(rank)])
        current_steps = torch.cat([current_steps, torch.zeros(1, dtype=torch.int64).to(rank)])
        desc_len = torch.cat([desc_len, torch.tensor([new_z_len]).to(rank)])
        bar_length = torch.cat([bar_length, torch.zeros(1, dtype=torch.int64).to(rank)])
        bar_no = torch.cat([bar_no, torch.tensor([1], dtype=torch.int64).to(rank)])
    terminated = torch.zeros(args.batch_size_ep, dtype=torch.bool).to(rank)
  cleanup()  

      
if __name__ == "__main__":
  # random.seed(42)
  world_size = 2
  mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)