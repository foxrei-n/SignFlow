from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch as th
from torch import Tensor
from loguru import logger
from scipy.linalg import sqrtm
from modules.reproduce_utils import seed_everything
from modules.dataset_csl import (
    CSLDataset,
    Stats,
    split_x,
    FACE_DIM 
)
from modules.tensor_utils import normalize, denormalize
from modules.rotation_utils import rotation_6d_to_axis_angle
from modules.render_utils_csl import render_smplx_sequences, get_vertices_all
from modules.fm import FlowMatchingModel

def parse_args():
    p = ArgumentParser()
    p.add_argument(
        '--chkpt-path', 
        type=str,
        help='Path to text-to-gesture model checkpoint.'
    )
    p.add_argument(
        '--n-timesteps',
        type=int,
        default=25,
        help='Number of denoising steps.'
    )
    p.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed'
    )
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    chkpt_path = '/logs/CFM_CSL/train_step_300000/chkpt.pth'

    seed_everything(args.seed)
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    dataset = CSLDataset('test') 
    stats = Stats(device=device)

    model = FlowMatchingModel.from_chkpt(args.chkpt_path)
    model.eval()
    model.to(device)
    dirpath = Path(args.chkpt_path).parent / 'vis'
    vis_dirpath = dirpath / f'steps_{args.n_timesteps}_seed_{args.seed}'
    vis_dirpath.mkdir(parents=True, exist_ok=True)
    sample = dataset.random_sample()
    
    faces, d6s = split_x(sample.x)
    poses = rotation_6d_to_axis_angle(d6s.contiguous().view(-1, 6)).view(d6s.size(0), -1)

    n_gen = 1
    normed_text_feats = sample.text_feat.to(device)
    text_size = th.tensor([len(sample.text_feat)])

    normed_text_feats = normed_text_feats.squeeze(0)
    

    cond_kwargs = dict(
        text_size=text_size.repeat(n_gen).to(device),
        text_feats=normed_text_feats[None].repeat(n_gen, 1, 1)
    )

    x_size = th.tensor([len(sample.x)]).repeat(n_gen).to(device)
    
    out = model.sample(
        x_size,
        cond_kwargs,
        n_timesteps=args.n_timesteps,
        progress=True
    )
    out_xs = denormalize(out, stats.x_mu, stats.x_sd)
    out_faces, out_d6s = split_x(out_xs)
    out_poses = rotation_6d_to_axis_angle(out_d6s.contiguous().view(-1, 6)).view(n_gen, out_d6s.size(1), -1)

    npz_path_list = []
    np.savez(
        str(vis_dirpath / f'gt.npz'),
        poses=poses.numpy(),
        trans=np.zeros((poses.size(0), 3)),
        expressions=faces.numpy(),
        betas=np.zeros((poses.size(0), 10)),
        mocap_framerate=30,
        model='SMPLX_NEUTRAL_2020',
        gender='neutral'
    )
    npz_path_list.append(str(vis_dirpath / f'gt.npz'))


    for i in range(n_gen):
        generated_pose = out_poses[i].cpu().numpy()
        np.savez(
            str(vis_dirpath / f'gen_{i}.npz'),
            poses=generated_pose,
            trans=np.zeros((out_poses[i].size(0), 3)),
            expressions=out_faces[i].cpu().numpy(),
            betas=np.zeros((out_poses[i].size(0), 10)),
            mocap_framerate=30,
            model='SMPLX_NEUTRAL_2020',
            gender='neutral'
        )
        npz_path_list.append(str(vis_dirpath / f'gen_{i}.npz'))
        
    render_smplx_sequences(
        npz_path_list,
        output_dir=vis_dirpath,
        output_name='vis',
        max_n_cols=1 + n_gen,
    )
