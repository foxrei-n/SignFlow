import os
import pickle
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import torch as th
from torch import Tensor
from transformers import MBartForConditionalGeneration, MBartTokenizer
from torch.nn.utils.rnn import pad_sequence
from .rotation_utils import axis_angle_to_rotation_6d
import pandas as pd
from loguru import logger

PROCESSED_DATA_PATH = Path('./data/train')
device = 'cuda' if th.cuda.is_available() else 'cpu'

tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="zh_CN") 
mbart = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25").to(device)

encoded_texts = []

for split in ["train", "val", "test"]:
    data_filepath = PROCESSED_DATA_PATH / f"{split}_data.pkl"
    encoded_text_filepath = PROCESSED_DATA_PATH / f"{split}_text_features.pkl"

    if encoded_text_filepath.exists():
        print(f"Skipping {split}, already encoded.")
        continue

    print(f"Processing {split} dataset...")

    with open(data_filepath, 'rb') as f:
        data = pickle.load(f)

    encoded_texts = []
    for text in tqdm(data['text_feat'], desc=f"Encoding {split} Text"):
        with th.no_grad():
            text_tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True).input_ids.to(device)
            text_embedding = mbart(text_tokens).encoder_last_hidden_state.cpu().numpy()
        encoded_texts.append(text_embedding)

    with open(encoded_text_filepath, 'wb') as f:
        pickle.dump(encoded_texts, f)

    print(f"Saved encoded text features for {split}!")

def get_data_filepath(split: str):
    return PROCESSED_DATA_PATH / f'{split}_data.pkl'

def get_stats_filepath():
    return PROCESSED_DATA_PATH / f'stats.pkl'

def split_x(x):
    face = x[..., :FACE_DIM]
    d6 = x[..., FACE_DIM:]
    return face, d6

@dataclass
class SequenceSample:
    x: Tensor 
    text_feat: Tensor 
    seq_id: str


@dataclass
class BatchData:
    x: Tensor 
    x_size: Tensor 
    text_feat: Tensor 
    text_size: Tensor 

@dataclass
class Stats:
    x_mu: Tensor
    x_sd: Tensor
    text_feat_mu: Tensor
    text_feat_sd: Tensor
    
    def __init__(self, filepath: Path = None, device='cpu'):
        if filepath is None:
            filepath = get_stats_filepath()
        with open(str(filepath), 'rb') as f:
            stats = pickle.load(f)
        self.x_mu = th.from_numpy(stats['x_mu']).float().to(device)
        self.x_sd = th.from_numpy(stats['x_sd']).float().to(device)

class H2SDataset:
    def __init__(self, split: str):
        data_filepath = get_data_filepath(split)
        text_feat_filepath = PROCESSED_DATA_PATH / f'{split}_text_features.pkl'

        logger.info(f'Loading {split} data from {data_filepath}')
        logger.info(f'Loading precomputed text features from {text_feat_filepath}')

        with open(data_filepath, 'rb') as f:
            self.data = pickle.load(f)

        with open(text_feat_filepath, 'rb') as f:
            self.text_feats = pickle.load(f)

        if split == 'train':
            x_mu = np.concatenate(self.data['x'], axis=0).mean(axis=0) 
            x_sd = np.concatenate(self.data['x'], axis=0).std(axis=0) + 1e-10

            stats = dict(   
                    x_mu=x_mu.astype(np.float32),
                    x_sd=x_sd.astype(np.float32)
                )
            stats_filepath = get_stats_filepath()
            with open(str(stats_filepath), 'wb') as f:
                    pickle.dump(stats, f)

        logger.info(f'{split} total sequences: {len(self.data["x"])}')
        total_frames = sum([len(m) for m in self.data['x']])
        logger.info(f'{split} total frames: {total_frames:,}')

        total_duration = total_frames / POSE_FPS / 3600
        logger.info(f'{split} total duration: {total_duration:.2f} hours')


    def __len__(self):
        length = len(self.data['x'])
        return length

    def __getitem__(self, idx: int):
        seq_idx = idx

        xs = self.data['x'][idx]  
        text_embedding = self.text_feats[idx]  
        sequence_ide = self.data['seq_id'][seq_idx]

        if len(text_embedding.shape) == 3:  
            text_embedding = text_embedding.squeeze(0)

        return SequenceSample(
            x=th.from_numpy(xs).float(),
            text_feat=th.from_numpy(text_embedding).float(),
            seq_id=sequence_ide
        )
    
    def random_sample(self, win_len: float = None):
        idx = np.random.randint(len(self.data['x']))
        return self.__getitem__(idx)

def custom_collate_fn(batch: list[SequenceSample]):
   
    xs_list = [item.x for item in batch]
    text_feats_list = [item.text_feat for item in batch] 

    xs_list_padded = pad_sequence(xs_list, batch_first=True)
    max_text_seq_len = max([t.shape[0] for t in text_feats_list])

    padded_text_feats_list = [
        th.cat([t, th.zeros(max_text_seq_len - t.shape[0], 1024)]) if t.shape[0] < max_text_seq_len else t
        for t in text_feats_list
    ]
    text_feats_padded = th.stack(padded_text_feats_list)

    x_size = th.tensor([len(x) for x in xs_list]).long()
    text_size = th.tensor([len(t) for t in text_feats_padded]).long()

    return BatchData(
        x=xs_list_padded,
        x_size=x_size,
        text_feat=text_feats_padded,
        text_size=text_size
    )
