"""Small DDP / logging / checkpoint helpers shared by the training scripts."""

import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist


def setup_ddp():
    """Returns (rank, world_size, device). Works with or without torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    return rank, world, torch.device("cuda", local)


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def all_reduce_(tensors):
    if dist.is_initialized():
        for t in tensors:
            dist.all_reduce(t)


def cosine_lr(step, total, base, warmup=1000, floor=0.05):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


class Logger:
    """Appends JSON lines to <out>/log.jsonl and mirrors a compact line to stdout."""

    def __init__(self, out):
        self.out = Path(out)
        self.t0 = time.time()

    def log(self, **kv):
        if not is_main():
            return
        kv = {k: (v.item() if torch.is_tensor(v) else v) for k, v in kv.items()}
        kv["time_h"] = round((time.time() - self.t0) / 3600, 3)
        with open(self.out / "log.jsonl", "a") as f:
            f.write(json.dumps(kv) + "\n")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kv.items()),
              flush=True)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def save_ckpt(path, model, opt=None, **extra):
    if not is_main():
        return
    state = {"model": unwrap(model).state_dict(), **extra}
    if opt is not None:
        state["opt"] = opt.state_dict()
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
