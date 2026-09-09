# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# 
# --------------------------------------------------------
# utilitary functions for CroCo
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
import math
import json
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

import torch
import torch.distributed as dist
from torch import inf

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, max_iter=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        len_iterable = min(len(iterable), max_iter) if max_iter else len(iterable)
        space_fmt = ':' + str(len(str(len_iterable))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for it,obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len_iterable - 1:
                eta_seconds = iter_time.global_avg * (len_iterable - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len_iterable, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len_iterable, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
            if max_iter and it >= max_iter:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len_iterable))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    nodist = args.nodist if hasattr(args,'nodist') else False 
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and not nodist:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True):
        self._scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm




def save_model(args, epoch, model_without_ddp, optimizer, loss_scaler, fname=None, best_so_far=None):
    output_dir = Path(args.output_dir)
    if fname is None: fname = str(epoch)
    checkpoint_path = output_dir / ('checkpoint-%s.pth' % fname)
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': loss_scaler.state_dict(),
        'args': args,
        'epoch': epoch,
    }
    if best_so_far is not None: to_save['best_so_far'] = best_so_far
    print(f'>> Saving model to {checkpoint_path} ...')
    save_on_master(to_save, checkpoint_path)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    args.start_epoch = 0
    best_so_far = None
    if args.resume is not None:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        print("Resume checkpoint %s" % args.resume)
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        args.start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        if 'best_so_far' in checkpoint:
            best_so_far = checkpoint['best_so_far']
            print(" & best_so_far={:g}".format(best_so_far))
        else:
            print("")
        print("With optim & sched! start_epoch={:d}".format(args.start_epoch), end='')
    return best_so_far

def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

def _replace(text, src, tgt, rm=''):
    """ Advanced string replacement.
    Given a text:
    - replace all elements in src by the corresponding element in tgt
    - remove all elements in rm
    """
    if len(tgt) == 1: 
        tgt = tgt * len(src)
    assert len(src) == len(tgt), f"'{src}' and '{tgt}' should have the same len"
    for s,t in zip(src, tgt):
        text = text.replace(s,t)
    for c in rm:
        text = text.replace(c,'')
    return text
    
def filename( obj ):
    """ transform a python obj or cmd into a proper filename. 
     - \1 gets replaced by slash '/'
     - \2 gets replaced by comma ','
    """
    if not isinstance(obj, str): 
        obj = repr(obj)
    obj = str(obj).replace('()','')
    obj = _replace(obj, '_,(*/\1\2','-__x%/,', rm=' )\'"')
    assert all(len(s) < 256 for s in obj.split(os.sep)), 'filename too long (>256 characters):\n'+obj
    return obj

def _get_num_layer_for_vit(var_name, enc_depth, dec_depth):
    if var_name in ("cls_token", "mask_token", "pos_embed", "global_tokens"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("enc_blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    elif var_name.startswith('decoder_embed') or var_name.startswith('enc_norm'): # part of the last black
        return enc_depth
    elif var_name.startswith('dec_blocks'):
        layer_id = int(var_name.split('.')[1])
        return enc_depth + layer_id + 1
    elif var_name.startswith('dec_norm'): # part of the last block
        return enc_depth + dec_depth
    elif any(var_name.startswith(k) for k in ['head','prediction_head']):
        return enc_depth + dec_depth + 1
    else:
        raise NotImplementedError(var_name)

def get_parameter_groups(model, weight_decay, layer_decay=1.0, skip_list=(), no_lr_scale_list=[]):
    """
    Build optimizer parameter groups with differential lr scaling for Dust3R + UniK3D hybrid.

    Keeps global LR scheduling (warmup + cosine) but multiplies each group's LR by a per-group lr_scale.
    When layer_decay < 1.0 (original MAE-style), the final lr_scale = (layer_decay_factor_for_layer * custom_scale).

    Groups & scales (relative to base LR):
        enc:                1.0   (encoder: patch_embed, enc_blocks, enc_norm)
        dec1:               1.0   (first decoder path: dec_blocks, decoder_embed, dec_norm)
        dec2:               1.0   (second decoder: dec_blocks2)
        head1_body:         1.0   (downstream_head1.dpt.* except final conv)
        head2_body:         1.5   (downstream_head2.dpt.* except final conv)
        head1_last:         1.5   (final 1x1 conv of head1)
        head2_last:         2.0   (final 1x1 conv of head2)
        adapter:            1.5   (modules whose name contains: 'angular', 'cam_adapter', 'camera_adapter')
        other:              1.0   (fallback)

    Head / adapter groups use lower weight decay (0.01) to reduce output collapse; others use `weight_decay`.

    Parameters that do not require grad are skipped.
    Bias / LayerNorm parameters still get weight_decay=0 (same logic as before).

    Returns:
        List[Dict]: each dict has keys: params, weight_decay, lr_scale
    """
    # If layer_decay < 1.0 we precompute per-layer base decay scale for ViT layers
    enc_depth, dec_depth = None, None
    layer_id_to_decay = {}
    if layer_decay < 1.0:
        enc_depth = getattr(model, 'enc_depth', None)
        dec_depth = getattr(model, 'dec_depth', 0 if not hasattr(model, 'dec_blocks') else model.dec_depth)
        if enc_depth is None:
            raise ValueError("Model must expose enc_depth for layer-wise decay.")
        num_layers = enc_depth + dec_depth
        # layer indices: [0 .. num_layers+1] similar to original
        base_scales = [layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]
        def _get_num_layer_for_vit(var_name):
            if var_name in ("cls_token", "mask_token", "pos_embed", "global_tokens"):
                return 0
            elif var_name.startswith("patch_embed"):
                return 0
            elif var_name.startswith("enc_blocks"):
                layer_id = int(var_name.split('.')[1])
                return layer_id + 1
            elif var_name.startswith('decoder_embed') or var_name.startswith('enc_norm'):
                return enc_depth
            elif var_name.startswith('dec_blocks'):
                layer_id = int(var_name.split('.')[1])
                return enc_depth + layer_id + 1
            elif var_name.startswith('dec_norm'):
                return enc_depth + dec_depth
            elif any(var_name.startswith(k) for k in ['head', 'prediction_head', 'downstream_head']):
                return enc_depth + dec_depth + 1
            else:
                # If a new module (adapter) doesn't match above, treat as last layer
                return enc_depth + dec_depth + 1
        # store for later lookup
        get_layer_id = _get_num_layer_for_vit
    else:
        get_layer_id = lambda _: 0
        base_scales = [1.0]  # unused

    # Custom group definitions
    CUSTOM_SCALES = dict(
        enc=1.0,
        dec1=1.0,
        dec2=1.0,
        head1_body=1.0,
        head2_body=1.5,
        head1_last=1.5,
        head2_last=2.0,
        adapter=1.5,
        other=1.0,
    )
    HEAD_DECAY = 0.01  # weight decay for heads & adapter

    # Prepare containers
    group_dict = {}  # name -> {params, weight_decay, lr_scale}
    group_dict_names = {}  # name -> {params, weight_decay, lr_scale}

    def _add_param(group_name, param, name, weight_decay_value, lr_scale_value):
        bias_string = ''
        if name.endswith(".bias") or "norm" in name:
            weight_decay_value = 0.0
            bias_string = '_bias'
            
        if group_name+bias_string not in group_dict:
            group_dict[group_name+bias_string] = {
                "params": [],
                "weight_decay": weight_decay_value,
                "lr_scale": lr_scale_value
            }
            group_dict_names[group_name+bias_string] = {
                "params": [],
                "weight_decay": weight_decay_value,
                "lr_scale": lr_scale_value
            }
        group_dict[group_name+bias_string]["params"].append(param)
        group_dict_names[group_name+bias_string]["params"].append(name)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine if no_decay (bias / LN)
        if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name or name in skip_list:
            this_wd = 0.0
        else:
            this_wd = weight_decay

        # Base layer decay scale (1.0 if layer_decay == 1)
        layer_id = get_layer_id(name)
        layer_decay_scale = base_scales[layer_id] if layer_decay < 1.0 else 1.0

        # Classify parameter into a custom group
        if name.startswith(('patch_embed', 'enc_blocks', 'enc_norm')):
            gname = 'enc' 
        elif name.startswith(('dec_blocks2',)):
            gname = 'dec2' 
        elif name.startswith(('dec_blocks', 'decoder_embed', 'dec_norm')):
            gname = 'dec1' 
        elif name.startswith('downstream_head1.dpt.head.4.'):
            gname = 'head1_last' 
        elif name.startswith('downstream_head2.dpt.head.4.'):
            gname = 'head2_last' 
        elif name.startswith('downstream_head1.dpt.'):
            gname = 'head1_body' 
        elif name.startswith('downstream_head2.dpt.'):
            gname = 'head2_body'
        elif any(tok in name for tok in ['angular', 'cam_adapter', 'camera_adapter']):
            gname = 'adapter'
        else:
            gname = 'other'

        custom_scale = CUSTOM_SCALES[gname]

        # If this parameter should NOT be lr-scaled (explicit in no_lr_scale_list), override to 1.0
        if name in no_lr_scale_list:
            final_scale = 1.0
        else:
            final_scale = layer_decay_scale * custom_scale

        # Heads / adapter get reduced weight decay unless already zero
        if gname in ['head1_body', 'head2_body', 'head1_last', 'head2_last', 'adapter'] and this_wd != 0.0:
            this_wd = HEAD_DECAY

        _add_param(gname, param, name, this_wd, final_scale)

    # Debug print similar to original (names & lr_scale only)
    printable = {
        k: {
            "params": len(v["params"]),
            "weight_decay": v["weight_decay"],
            "lr_scale": v["lr_scale"],
        } for k, v in group_dict.items()
    }
    print("Custom param groups (differential LR) =\n" + json.dumps(printable, indent=2))
    printable = {
        k: {
            "params": v["params"],
            "weight_decay": v["weight_decay"],
            "lr_scale": v["lr_scale"],
        } for k, v in group_dict_names.items()
    }
    print("Custom param groups (differential LR) =\n" + json.dumps(printable, indent=2))

    return list(group_dict.values())



def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
            
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
            
    return lr
