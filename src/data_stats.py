import math
import random
from collections import namedtuple, deque
from itertools import count
import re
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import argparse
import os
import glob
import torch
import random
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import BertAttention

from models.vae import VqVaeModule
from models.seq2seq import Seq2SeqModule
from datasets import MidiDataset, SeqCollator
from utils import medley_iterator
from input_representation import remi2midi
from vocab import DescriptionVocab

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

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
TEMP = 0.8

def parse_args():
  parser = argparse.ArgumentParser()
  # parser.add_argument('--model', type=str, required=True, help="Model name (one of 'figaro', 'figaro-expert', 'figaro-learned', 'figaro-no-inst', 'figaro-no-chord', 'figaro-no-meta')")
  # parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint")
  parser.add_argument('--model', type=str, default="figaro-expert")
  parser.add_argument('--checkpoint', type=str, default="./checkpoints/figaro-expert.ckpt")
  parser.add_argument('--vae_checkpoint', type=str, default=None, help="Path to the VQ-VAE model checkpoint (optional)")
  parser.add_argument('--lmd_dir', type=str, default='./lmd_full', help="Path to the root directory of the LakhMIDI dataset")
  parser.add_argument('--max_iter', type=int, default=16_000)
  parser.add_argument('--max_bars', type=int, default=32)
  parser.add_argument('--batch_size', type=int, default=1)
  parser.add_argument('--verbose', type=int, default=1)
  parser.add_argument('--output_dir', type=str, default='./samples', help="Path to the output directory")
  parser.add_argument('--max_n_files', type=int, default=-1)


  args = parser.parse_args()
  return args

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


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
  state_dict = {k: v for k, v in state_dict.items() if not k.endswith('embeddings.position_ids')}
  try:
    # succeeds for checkpoints trained with transformers>4.13.0
    model.load_state_dict(state_dict)
  except RuntimeError:
    # work around a breaking change introduced in transformers==4.13.0, which fixed the position_embedding_type of cross-attention modules "absolute"
    config = model.transformer.decoder.bert.config
    for layer in model.transformer.decoder.bert.encoder.layer:
      layer.crossattention = BertAttention(config, position_embedding_type=config.position_embedding_type)
    model.load_state_dict(state_dict)
  model.freeze()
  model.eval()
  return model


def load_model(checkpoint, device='auto'):
  if device == 'auto':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = load_old_or_new_checkpoint(Seq2SeqModule, checkpoint)
  model.to(device)

  return model

# def load_model(checkpoint, vae_checkpoint=None, device='auto'):
#   if device == 'auto':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#   vae_module = None
#   if vae_checkpoint:
#     vae_module = load_old_or_new_checkpoint(VqVaeModule, vae_checkpoint)
#     vae_module.cpu()

#   model = load_old_or_new_checkpoint(Seq2SeqModule, checkpoint)
#   model.to(device)

#   return model, vae_module


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
      

# n_actions = env.action_space.n
# Get the number of state observations
# state, info = env.reset()
# n_observations = len(state)
args = parse_args()

policy_net = load_model(args.checkpoint)
target_net = load_model(args.checkpoint)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

      

def select_action(state, model):
  """_summary_

  Args:
      state (dict{'input_ids', 'bar_ids', 'position_ids', 'description', 'desc_bar_ids'}): 고정되지 않은 사이즈

  Returns:
      next_token_ids (tensor(int)): (batch_size, 1)
      next_tokens (tensor(str)): (batch_size, 1)
  """
  x = state['input_ids'].to(model.device)
  bar_ids = state['bar_ids'].to(model.device)
  position_ids = state['position_ids'].to(model.device)
  z = state['description'].to(model.device)
  desc_bar_ids = state['desc_bar_ids'].to(model.device)
  idx = x.shape[1]-1
  with torch.no_grad():
    encoder_hidden_states = model.encode(z, desc_bar_ids)
    logits = model.decode(x, bar_ids=bar_ids, position_ids=position_ids, encoder_hidden_states=encoder_hidden_states)
    logits = logits[:, idx] / TEMP
    # print(logits.size())
    pr = F.softmax(logits, dim=-1)
    pr = pr.view(-1, pr.size(-1))
    next_token_ids = torch.multinomial(pr, 1).view(-1).to(x.device)
    next_tokens = model.vocab.decode(next_token_ids)
    return next_token_ids, next_tokens
  
MAX_BARS = 24
  
def step(state, 
  action, 
  z, 
  desc_bar_ids,
  model,
  vocab, 
  terminated,
  events_full,
  bar_ids_full,
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
  pad_token_id = policy_net.vocab.to_i(pad_token)
  eos_token_id = policy_net.vocab.to_i(eos_token)
  x = state['input_ids']
  bar_ids = state['bar_ids']
  position_ids = state['position_ids']
  z_ = state['description']
  desc_bar_ids_ = state['desc_bar_ids']
  next_token_ids, next_tokens = action
  i = x.shape[1]-1
  batch_size = x.shape[0]
  is_done = terminated
  context_size = model.context_size
  curr_bars = bar_ids[:, 0]
  
  next_bars = torch.tensor([1 if f'{BAR_KEY}_' in token else 0 for token in next_tokens], dtype=torch.int)
  next_bar_ids = bar_ids[:, i].clone() + next_bars

  next_positions = [f"{POSITION_KEY}_0" if f'{BAR_KEY}_' in token else token for token in next_tokens]
  next_positions = [int(token.split('_')[-1]) if f'{POSITION_KEY}_' in token else None for token in next_positions]
  next_positions = [pos if next_pos is None else next_pos for pos, next_pos in zip(position_ids[:, i], next_positions)]
  next_position_ids = torch.tensor(next_positions, dtype=torch.int)

  is_done.masked_fill_((next_token_ids == eos_token_id).all(dim=-1), True)
  is_done.masked_fill_(next_bar_ids >= MAX_BARS + 1, True)
  
  # print(next_token_ids.shape)
  # print(is_done)
  
  next_token_ids[is_done] = pad_token_id

  x = torch.cat([x, next_token_ids.clone().unsqueeze(1)], dim=1)
  bar_ids = torch.cat([bar_ids, next_bar_ids.unsqueeze(1)], dim=1)
  position_ids = torch.cat([position_ids, next_position_ids.unsqueeze(1)], dim=1)  
  
  #x, bar_ids, position_ids 자르기, z, desc_bar_ids scrolling
  x = x[:, -context_size:].to(model.device)
  bar_ids = bar_ids[:, -context_size:].to(model.device)
  position_ids = position_ids[:, -context_size:].to(model.device)
  
  next_bars = bar_ids[:, 0]
  bars_changed = not (next_bars == curr_bars).all()
  
  if bars_changed:
    z_ = torch.zeros(batch_size, context_size, dtype=torch.int)
    desc_bar_ids_ = torch.zeros(batch_size, context_size, dtype=torch.int)

    for j in range(batch_size):
      curr_bar = bar_ids[j, 0]
      indices = torch.nonzero(desc_bar_ids[j] == curr_bar)
      if indices.size(0) > 0:
        idx = indices[0, 0]
      else:
        idx = z.size(1) - 1

      offset = min(context_size, z.size(1) - idx)

      z_[j, :offset] = z[j, idx:idx+offset]
      desc_bar_ids_[j, :offset] = desc_bar_ids[j, idx:idx+offset]

    z_, desc_bar_ids_ = z_.to(model.device), desc_bar_ids_.to(model.device)
  
  reward = torch.zeros(batch_size, dtype=torch.float32)
  
  for j in range(batch_size):
    if f'{BAR_KEY}_' in action[1][j] and int(action[1][j].split('_')[-1]) != 1:
      current_bar = bar_ids[j][-2]
      
      events_indices = torch.nonzero(bar_ids_full[j] == current_bar)
      desc_indices = torch.nonzero(desc_bar_ids[j] == current_bar)
      curr_bar_seq = events_full[j][events_indices]
      curr_bar_desc = vocab.decode(z[j][desc_indices])
      curr_bar_seq = [model.vocab.decode(x)[0] for x in curr_bar_seq]
      meta_sample = get_meta_sample(curr_bar_seq)
      meta_org = get_meta_org(curr_bar_desc)
      
      print("meta sample: ")
      print(meta_sample)
      print("meta org: ")
      print(meta_org)      

  return {'input_ids' : x,
          'bar_ids' : bar_ids,
          'position_ids' : position_ids,
          'description' : z_,
          'desc_bar_ids' : desc_bar_ids_}, reward, is_done
  
  
def get_positions_per_bar(bar_seq):
  time_sig = [token.split("_")[-1] for token in bar_seq if token.startswith(f'{TIME_SIGNATURE_KEY}_')][0]
  numerator, denominator = [int(x) for x in time_sig.split("/")]
  # print(numerator, denominator)
  quarters_per_bar = 4 * numerator / denominator
  positions_per_bar = int(DEFAULT_POS_PER_QUARTER * quarters_per_bar)
  return positions_per_bar

def get_meta_org(bar_desc):
  meta = dict()
  time_signature = [x.split("_")[-1] for x in bar_desc if f'{TIME_SIGNATURE_KEY}_'in x]
  if len(time_signature) > 0:
    meta['time_signature'] = time_signature[0]
  else:
    meta['time_signature'] = None
  
  instruments = set([x.split("_")[-1] for x in bar_desc if f'{INSTRUMENT_KEY}_'in x])
  if len(instruments) > 0:
    meta['instruments'] = instruments
  else:
    meta['instruments'] = None
  
  chords = set([x.split("_")[-1] for x in bar_desc if f'{CHORD_KEY}_'in x])
  if len(chords) > 0:
    meta['chords'] = chords
  else:
    meta['chords'] = None
  
  note_density = [x.split("_")[-1] for x in bar_desc if f'{NOTE_DENSITY_KEY}_'in x]
  if len(note_density) > 0:
    meta['note_density'] = note_density[0]
  else:
    meta['note_density'] = 0
    
  mean_pitch = [x.split("_")[-1] for x in bar_desc if f'{MEAN_PITCH_KEY}_'in x]
  if len(mean_pitch) > 0:
    meta['mean_pitch'] = mean_pitch[0]
  else:
    meta['mean_pitch'] = 0
    
  return meta
    
  
def get_meta_sample(bar_seq):
  meta = dict()
  positions_per_bar = get_positions_per_bar(bar_seq)
  notes = []
  for i in range(len(bar_seq) - 4):
    notes.append(bar_seq[i:i+5])
  notes = [note for note in notes if (f'{POSITION_KEY}_'in note[0] and 
                                      f'{INSTRUMENT_KEY}_'in note[1] and 
                                      f'{PITCH_KEY}_'in note[2] and 
                                      f'{VELOCITY_KEY}_'in note[3] and 
                                      f'{DURATION_KEY}_'in note[4])]
  time_signature = [ts.split('_')[-1] for ts in bar_seq if f'{TIME_SIGNATURE_KEY}_'in ts]
  if len(time_signature) > 0:
    meta['time_signature'] = time_signature[0]
  else:
    meta['time_signature'] = None
    
  instruments = set([note[1].split("_")[-1] for note in notes])
  if len(instruments) > 0:
    meta['instruments'] = instruments
  else: 
    meta['instruments'] = None
    
  chords = set([c.split("_")[-1] for c in bar_seq if f'{CHORD_KEY}_'in c])
  if len(chords) > 0:
    meta['chords'] = chords
  else:
    meta['chords'] = None
  
  if len(notes) > 0:
    meta['note_density'] = np.argmin(abs(DEFAULT_NOTE_DENSITY_BINS - len(notes)/positions_per_bar))
    # mean_velocity = np.mean([min(127, DEFAULT_VELOCITY_BINS[int(note[3].split("_")[-1])]) for note in notes])
    # mean_velocity = np.argmin(abs(DEFAULT_MEAN_VELOCITY_BINS-mean_velocity))
    mean_pitch = np.mean([int(note[2].split("_")[-1]) for note in notes])
    meta['mean_pitch'] = np.argmin(abs(DEFAULT_MEAN_PITCH_BINS-mean_pitch))
  else:
    meta['note_density'] = 0
    # mean_velocity = 0
    meta['mean_pitch'] = 0
    
  return meta
  
  
  



if __name__ == '__main__':
  args = parse_args() 
  
  model = load_model(args.checkpoint)
  print("context size: ", model.context_size)
  midi_files = glob.glob(os.path.join(args.lmd_dir, '**/*.mid'), recursive=True)
  dm = model.get_datamodule(midi_files)
  dm.setup('train')
  midi_files = dm.train_ds.files
  random.shuffle(midi_files)
  if args.max_n_files > 0:
    midi_files = midi_files[:args.max_n_files]  
  print(len(midi_files))
  vocab = DescriptionVocab()
  dataset = MidiDataset(
    midi_files,
    max_len=-1,
    description_flavor=model.description_flavor,
    max_bars=model.context_size,
  )

  coll = SeqCollator(context_size=-1)
  dl = DataLoader(dataset, batch_size=1, collate_fn=coll)

  
  initial_context = 1
  j = 0

  note_density = []
  for batch in dl:
    j+=1
    print(f'{j}/137673')
    desc = vocab.decode(batch['description'][0])
    for token in desc:
      if f'{NOTE_DENSITY_KEY}_' in token:
        note_density.append(int(token.split("_")[-1]))
    if j >= 1000:
      
  note_density = np.array(note_density)
  count = np.bincount(note_density)
  print(count)
  
  

      
      
