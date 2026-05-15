from pathlib import Path
from typing import Callable
import torch as th
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .model import Estimator
from .optimal_transport import OTPlanSampler
from .model_utils import load_state_dict, add_noise_to_embeddings

    
class FlowMatchingModel(nn.Module):
    def __init__(
        self,
        x_dim: int,
        sigma_min: float = 1e-4,
        ot_plan: str = 'exact',
        **estimator_kwargs
    ):
        super().__init__()
        self.estimator = Estimator(x_dim, **estimator_kwargs)
        self.x_dim = x_dim
        self.ot_plan = ot_plan
        self.sigma_min = sigma_min
        self.ot_planner = OTPlanSampler(ot_plan)
        
    def compute_loss(
        self,
        xs: Tensor,
        x_size: Tensor,
        cond_kwargs: dict[str, Tensor],
        t: Tensor = None
    ):
        x1 = xs
        z = th.randn_like(x1)
        cond_list = [x_size] + list(cond_kwargs.values())
        z, x1, _, cond_list = self.ot_planner.sample_plan_with_multi_labels(
            z, x1, y0=None, y1=cond_list
        )
        x_size = cond_list[0]
        cond_kwargs = {k: v for k, v in zip(cond_kwargs.keys(), cond_list[1:])}
        if t is None:
            t = th.rand(x1.size(0), 1, 1, device=x1.device, dtype=x1.dtype)
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1 g
        u = x1 - (1 - self.sigma_min) * z
        out = self.estimator(y, x_size, t.flatten(), **cond_kwargs)

        loss = F.mse_loss(out, u, reduction='none').mean(-1) 
        mask = th.arange(x1.size(1), device=x1.device).expand(x1.size(0), -1) < x_size[:, None] 
        loss = (loss * mask).sum(-1) / mask.sum(-1) 
        loss = loss.mean() 

        return loss
    
    def solve_euler(
        self, 
        x: Tensor, 
        x_size: Tensor, 
        cond_kwargs: dict[str, Tensor], 
        t_span: Tensor,
        modulation_func: Callable = None,
        progress: bool = False
    ):
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        
        if progress:
            iter = tqdm(range(1, len(t_span)), desc='Solving', leave=False, ascii=True, dynamic_ncols=True)
        else:
            iter = range(1, len(t_span))

        for step in iter:
            t_in = t * th.ones(x.size(0), device=x.device)
            with th.no_grad():
                dpsi_dt = self.estimator(x, x_size, t_in, **cond_kwargs)
                              
            if modulation_func:
                dpsi_dt = modulation_func(dpsi_dt, x, x_size, t, dt)
            x = x + dt * dpsi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return x.detach()

    def sample(
        self, 
        x_size: Tensor, 
        cond_kwargs: dict[str, Tensor], 
        n_timesteps: int = 50, 
        temperature: float = 1.0, 
        progress: bool = False    
    ) -> Tensor:
        device = next(self.parameters()).device
        z = th.randn(x_size.size(0), x_size.max(), self.x_dim, device=device) * temperature
        t_span = th.linspace(0, 1, n_timesteps + 1, device=device)
        return self.solve_euler(
            z, 
            x_size, 
            cond_kwargs, 
            t_span, 
            progress=progress
        )
    
    @staticmethod
    def from_chkpt(chkpt_path: Path, x_dim: int = 433):
        chkpt = th.load(chkpt_path, map_location='cpu')
        model_kwargs = chkpt['cfg']['model']
        if 'x_dim' not in model_kwargs:
            model_kwargs['x_dim'] = x_dim
            
        if 'attn_dropout' in model_kwargs:
            prob = model_kwargs.pop('attn_dropout')
            model_kwargs['self_attn_dropout'] = prob
            model_kwargs['cross_attn_dropout'] = prob
            
        if 'dropout' in model_kwargs:
            prob = model_kwargs.pop('dropout')
            model_kwargs['audio_encoder_dropout'] = prob
            model_kwargs['ff_dropout'] = prob
            model_kwargs['self_attn_dropout'] = prob
            model_kwargs['cross_attn_dropout'] = prob
        
        model = FlowMatchingModel(**model_kwargs)
        if 'avg_model_state' in chkpt:
            load_state_dict(model, chkpt['avg_model_state'])
        else:
            model.load_state_dict(chkpt['model_state'])
            
        return model