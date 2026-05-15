import argparse
import json
from pathlib import Path
import shutil
import wandb
import numpy as np
import torch as th
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
from modules.reproduce_utils import seed_everything
from modules.dataset_csl import H2SDataset, Stats, custom_collate_fn
from modules.model_utils import check_grad_and_clip_, count_params
from modules.tensor_utils import normalize
from modules.fm import FlowMatchingModel
import torch.multiprocessing as mp
import math



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--run-name', type=str, default='none')
    p.add_argument('--sigma-min', type=float, default=1e-4)
    p.add_argument('--noise-level', type=float, default=0.01)
    p.add_argument('--ot-plan', type=str, default='exact', choices=['exact'])
    p.add_argument('--dim', type=int, default=64, help='dim for each head')
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--down-ts', type=int, default=2)
    p.add_argument('--attn-type', type=str, default='mha', choices=['mha'])
    p.add_argument('--n-attn-blocks', type=int, default=1)
    p.add_argument('--num-mid-blocks', type=int, default=6)
    p.add_argument('--emb-dim-expansion', type=int, default=1)
    p.add_argument('--disable-mem-mean', action='store_true')
    p.add_argument('--self-attn-dropout', type=float, default=0.1)
    p.add_argument('--cross-attn-dropout', type=float, default=0.1)
    p.add_argument('--ff-dropout', type=float, default=0.1)
    p.add_argument('--total-train-steps', type=int, default=300_000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--micro-batch-size', type=int, default=None)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup_steps', type=int, default=10000, help='Warm-up steps for LR scheduler')
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--ema-rate', type=float, default=0.999)
    p.add_argument('--use-amp', action='store_true')
    p.add_argument('--num-workers', type=int, default=32) 
    p.add_argument('--log-interval', type=int, default=10)
    p.add_argument('--save-interval', type=int, default=10_000)
    p.add_argument('--eval-interval', type=int, default=5_000)
    p.add_argument('--keep-interval', type=int, default=50_000)
    p.add_argument('--val-n-samples', type=int, default=2048)
    
    args = p.parse_args()
    if args.micro_batch_size is None:
        args.micro_batch_size = args.batch_size
    args.val_n_batches = args.val_n_samples // args.micro_batch_size
        
    return args

def main():
    args = parse_args()
    seed_everything(args.seed) 
    log_dirpath = Path(f'./log/{args.run_name}') 
    log_dirpath.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_dirpath / 'train_{time}.log'))
    cfg = dict(
        seed=args.seed,
        device=args.device,
        training=dict(
            total_train_steps=args.total_train_steps,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            lr=args.lr,
            max_grad_norm=args.max_grad_norm,
            ema_rate=args.ema_rate, 
            use_amp=args.use_amp
        ),
        model=dict(
            sigma_min=args.sigma_min,
            ot_plan=args.ot_plan,
            dim=args.dim,
            heads=args.heads,
            down_ts=args.down_ts,
            attn_type=args.attn_type,
            n_attn_blocks=args.n_attn_blocks,
            num_mid_blocks=args.num_mid_blocks,
            emb_dim_expansion=args.emb_dim_expansion,
            disable_mem_mean=args.disable_mem_mean,
            self_attn_dropout=args.self_attn_dropout,
            cross_attn_dropout=args.cross_attn_dropout,
            ff_dropout=args.ff_dropout,
        )
    )
    dataset = H2SDataset('train')
    val_dataset = H2SDataset('val')
    stats = Stats()
    cfg['model']['x_dim'] = stats.x_mu.shape[0]
    with open(str(log_dirpath / 'cfg.json'), 'w', encoding='utf-8') as fp:
        json_dumps_str = json.dumps(cfg, indent=4)
        print(json_dumps_str, file=fp)
    logger.info(f'Configs: {json_dumps_str}')

    assert args.batch_size % args.micro_batch_size == 0
    grad_accum_steps = args.batch_size // args.micro_batch_size
    n_accumed_grad = 0

    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )
    model = FlowMatchingModel(**cfg['model'])
    model.to(args.device)
    model.train()
    total_num_params, num_params = count_params(model)
    logger.info(f'Trainable parameters: {num_params:,}')
    cfg['total_num_params'] = total_num_params
    cfg['num_params'] = num_params
    avg_model = th.optim.swa_utils.AveragedModel(
        model,
        multi_avg_fn=th.optim.swa_utils.get_ema_multi_avg_fn(
            decay=args.ema_rate
        )
    )
    optimizer = th.optim.RAdam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr
    )

    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        progress = float(current_step - args.warmup_steps) / float(max(1, args.total_train_steps - args.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    grad_scaler = th.cuda.amp.GradScaler(enabled=args.use_amp)
    train_step = 0
    saved = False
    evaluated = False
    invalid_grad_counter = 0
    run_id = wandb.util.generate_id()
    wandb.init(
        project='test_experiment_0',
        group=args.run_name,
        name=f'{args.run_name}_runId_{run_id}',
        id=run_id,
        config=cfg,
        dir=str(log_dirpath),
        resume='never'
    )
    wandb.define_metric('train_step')
    wandb.define_metric('samples_seen')
    wandb.define_metric('train/*', step_metric='train_step')
    wandb.define_metric('val/*', step_metric='train_step')

    if args.resume_from_step is not None:
        logger.info(f'Resuming from train_step {args.resume_from_step}')
        resume_chkpt = th.load(
            str(log_dirpath / f'train_step_{args.resume_from_step}/chkpt.pth'), 
            map_location='cpu' 
        )
        model.load_state_dict(resume_chkpt['model_state'])
        avg_model.load_state_dict(resume_chkpt['avg_model_state'])
        optimizer.load_state_dict(resume_chkpt['optimizer_state'])
        grad_scaler.load_state_dict(resume_chkpt['grad_scaler_state'])
        train_step = resume_chkpt['train_step']
    run_steps = args.total_train_steps - train_step
    logger.info(f'Total epochs: {run_steps * grad_accum_steps / len(dataloader):.2f}')
    
    pbar = tqdm(
        total=run_steps,
        dynamic_ncols=True,
        desc='Training'
    )
    while True:
        for batch in dataloader:
            if not saved and train_step % args.save_interval == 0:  
                chkpt_dirpath = log_dirpath / f'train_step_{train_step}/'
                chkpt_dirpath.mkdir(parents=True, exist_ok=True)
                chkpt = dict(
                    model_state=model.state_dict(),
                    avg_model_state=avg_model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    grad_scaler_state=grad_scaler.state_dict(),
                    train_step=train_step,
                    cfg=cfg
                )
                th.save(chkpt, str(chkpt_dirpath / 'chkpt.pth'))
                saved = True
            num_evaled_batches = 0
            if not evaluated and train_step % args.eval_interval == 0:
                avg_model.eval()
                val_fm_loss = 0.0
                for val_batch_idx, val_batch in tqdm(
                    enumerate(val_dataloader),
                    total=args.val_n_batches,
                    desc='Validation', 
                    dynamic_ncols=True, 
                    leave=False
                ):
                    normed_xs = normalize(val_batch.x, stats.x_mu, stats.x_sd)
                    normed_text_feats = val_batch.text_feat
                    cond_kwargs = dict(
                        text_size=val_batch.text_size.to(args.device),
                        text_feats=normed_text_feats.to(args.device)
                    )
                    with th.no_grad():
                        with th.cuda.amp.autocast(enabled=args.use_amp, dtype=th.bfloat16):
                            val_fm_loss += avg_model.module.compute_loss(
                                normed_xs.to(args.device), 
                                val_batch.x_size.to(args.device), 
                                cond_kwargs
                            )

                    num_evaled_batches += 1
                    
                    if val_batch_idx == args.val_n_batches - 1:
                        break
                        
                wandb.log({
                    'train_step': train_step,
                    'val/fm_loss': val_fm_loss / num_evaled_batches
                })
                evaluated = True
                avg_model.train()
            if train_step >= args.total_train_steps:
                logger.info('Total train steps reached. Stop training.')
                break
                
            normed_xs = normalize(batch.x, stats.x_mu, stats.x_sd)
            normed_text_feats = batch.text_feat
            
            cond_kwargs = dict(
                text_size=batch.text_size.to(args.device),
                text_feats = normed_text_feats.to(args.device) 
            )
            with th.cuda.amp.autocast(enabled=args.use_amp, dtype=th.bfloat16):
                fm_loss = model.compute_loss(
                    normed_xs.to(args.device), 
                    batch.x_size.to(args.device), 
                    cond_kwargs
                )
                loss = fm_loss

            grad_scaler.scale(loss / grad_accum_steps).backward()
            n_accumed_grad += 1
            
            if n_accumed_grad == grad_accum_steps:
                grad_scaler.unscale_(optimizer)
                grad_is_valid, grad_norm = check_grad_and_clip_(
                    model, 
                    model_name='model',
                    max_norm=args.max_grad_norm,
                    verbose=0, 
                    set_none_to_zero=True
                )
                grad_scaler.step(optimizer)
                scheduler.step() 
                grad_scaler.update()               
                n_accumed_grad = 0
                if grad_is_valid:
                    invalid_grad_counter = 0
                else:
                    invalid_grad_counter += 1
                    optimizer.zero_grad() 
                    logger.warning(f'Model inf grad. Skipping update. Train step={train_step}.')
                    if invalid_grad_counter > 100:
                        wandb.alert(
                            title="Invalid grad",
                            text=f"Invalid grad for 100 steps. Train step={train_step}.",   
                            level=wandb.AlertLevel.ERROR,
                        )
                        raise RuntimeError(f'Invalid grad for 100 steps. Train step={train_step}.')
                    continue
                
                avg_model.update_parameters(model)
                optimizer.zero_grad()
                if train_step % args.log_interval == 0:
                    log_dict = {
                        'train_step': train_step,
                        'train/loss': loss.item(),
                        'train/fm_loss': fm_loss.item(),
                        'train/learning_rate': optimizer.param_groups[0]['lr'],
                        'train/grad_scale': grad_scaler.get_scale(),
                        'train/grad_norm': grad_norm
                    }
                    wandb.log(log_dict)
                train_step += 1
                pbar.update(1)
                saved = False
                evaluated = False
        else:
            continue
        break
    for chkpt_dirpath in log_dirpath.glob('train_step_*'):
        if int(chkpt_dirpath.name.split('_')[-1]) % args.keep_interval != 0:
            shutil.rmtree(chkpt_dirpath)
            

if __name__ == '__main__':
    main()
        