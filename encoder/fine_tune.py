import math
import os
import logging
import gc
import warnings
import time
import psutil
import numpy as np
import torch
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from dataset_processor import DatasetProcessor
from triplet_loss import TripletLoss
from wrapper import EncoderWrapper
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel._functions")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def setup_logging(log_file="training.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )


def configure_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(0.98, device=i)
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} - Total Memory: {total_memory:.2f}GB")


def log_memory_usage(device, prefix=""):
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        logger.info(f"{prefix} GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
    ram = psutil.virtual_memory()
    logger.info(f"{prefix} System RAM: {ram.used / (1024 ** 3):.2f}GB used, {ram.available / (1024 ** 3):.2f}GB available ({ram.percent}%)")


def empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def choose_model_load_path(dataset_id: int) -> str:
    if dataset_id == 1:
        return "./models/albertina-1b5"
    prev_path = f"./results/finetuned-albertina-1b5-triplet-loss-dataset-{dataset_id - 1}"
    return prev_path if os.path.isdir(prev_path) else "./models/albertina-1b5"


def save_full_wrapper(model, tokenizer, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.encoder.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    head_state = {
        "pooling_strategy": getattr(model_to_save, "pooling_strategy", "multihead"),
        "noise_std": getattr(model_to_save, "noise_std", 0.0),
        "projection": model_to_save.projection.state_dict(),
        "attention_heads": model_to_save.attention_heads.state_dict() if hasattr(model_to_save, "attention_heads") else None,
    }
    torch.save(head_state, os.path.join(save_dir, "wrapper_head.pt"))
    logger.info(f"✅ Model and tokenizer saved to {save_dir}")


def restore_full_wrapper(model, load_dir: str, device):
    head_path = os.path.join(load_dir, "wrapper_head.pt")
    if os.path.exists(head_path):
        state = torch.load(head_path, map_location=device)
        if hasattr(model, "attention_heads") and state.get("attention_heads") is not None:
            model.attention_heads.load_state_dict(state["attention_heads"])
        model.projection.load_state_dict(state["projection"])
        if "noise_std" in state:
            model.noise_std = state["noise_std"]


def calibrate_margins_for_dataset_rank0_only(rank, world_size, model, train_dataset,
                                             base_initial_cap=0.20, base_final_floor=0.05, final_ratio=0.60,
                                             probe_batches=8, batch_size=32):
    device = torch.device(f'cuda:{rank}')
    if rank == 0:
        from torch.utils.data import DataLoader, SequentialSampler
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=SequentialSampler(train_dataset),
            pin_memory=True,
            num_workers=0,   # keep it simple to avoid worker flakiness here
            drop_last=False,
        )
        model.eval()
        gaps = []
        with torch.no_grad():
            it = iter(loader)
            for _ in range(probe_batches):
                try:
                    batch = next(it)
                except StopIteration:
                    break
                a = {k: v.to(device) for k, v in batch['anchor'].items()}
                p = {k: v.to(device) for k, v in batch['positive'].items()}
                n = {k: v.to(device) for k, v in batch['negative'].items()}
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                    va = model(a['input_ids'], a['attention_mask'])
                    vp = model(p['input_ids'], p['attention_mask'])
                    vn = model(n['input_ids'], n['attention_mask'])
                pos_sim = (va * vp).sum(dim=1).clamp(-1 + 1e-8, 1 - 1e-8)
                neg_sim = (va * vn).sum(dim=1).clamp(-1 + 1e-8, 1 - 1e-8)
                gaps.append((pos_sim - neg_sim).detach().float().cpu())
        if gaps:
            gaps_all = torch.cat(gaps)
            init = torch.quantile(gaps_all, 0.25).item()
        else:
            init = 0.12
        init = max(0.06, min(init, base_initial_cap))
        fin = max(base_final_floor, min(final_ratio * init, 0.16))
        if rank == 0:
            logging.info(f"[Calib] Rank0 margins: initial={init:.3f}, final={fin:.3f}")
        margins = torch.tensor([init, fin], dtype=torch.float32, device=device)
    else:
        margins = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device)

    if torch.distributed.is_initialized():
        torch.distributed.broadcast(margins, src=0)

    return float(margins[0].item()), float(margins[1].item())


def train_distributed(rank, world_size, num_epochs=10, start_index=1, num_datasets=5):
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    configure_gpu_memory()

    for dataset_id in range(start_index, start_index + num_datasets):
        dataset_name = f"dataset_{dataset_id}.json"
        load_path = choose_model_load_path(dataset_id)
        if rank == 0:
            logger.info(f"Starting training on {dataset_name} using model: {load_path}")

        tokenizer = AutoTokenizer.from_pretrained(load_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = EncoderWrapper(model_name=load_path).to(device)
        restore_full_wrapper(model, load_path, device)
        model.encoder.gradient_checkpointing_enable()

        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=512,
            static_graph=True
        )

        projection_params = list(model.module.projection.parameters())
        encoder_params = [p for n, p in model.named_parameters() if "projection" not in n]

        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "weight_decay": 0.01, "lr": 2e-5},
                {"params": projection_params, "weight_decay": 0.01, "lr": 5e-4},
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        train_dataset = DatasetProcessor(tokenizer, split="train", max_length=512, dataset=dataset_name, hard_negatives=False)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=32,
            persistent_workers=True,
            prefetch_factor=8,
            drop_last=True,
        )

        accumulation_steps = 4
        opt_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
        total_opt_steps = opt_steps_per_epoch * num_epochs
        warmup_steps = max(50, int(0.08 * total_opt_steps))

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_opt_steps
        )

        initial_margin, final_margin = calibrate_margins_for_dataset_rank0_only(
            rank, world_size, model.module, train_dataset,
            base_initial_cap=0.20, base_final_floor=0.05, final_ratio=0.60,
            probe_batches=8, batch_size=32
        )

        if rank == 0:
            logger.info(f"Margin schedule for {dataset_name}: {initial_margin:.3f} → {final_margin:.3f}")

        run_training_loop(
            rank, world_size, model, train_loader, train_sampler,
            num_epochs, initial_margin, final_margin, optimizer, scheduler, dataset_id
        )

        if rank == 0:
            save_dir = f"./results/finetuned-albertina-1b5-triplet-loss-dataset-{dataset_id}"
            save_full_wrapper(model, tokenizer, save_dir)

        if dist.is_initialized():
            dist.barrier()
        torch.cuda.empty_cache()
        gc.collect()

    dist.destroy_process_group()


def _probe_epoch_batch_sims(model, batch, device):
    model.eval()
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        a = {k: v.to(device) for k, v in batch['anchor'].items()}
        p = {k: v.to(device) for k, v in batch['positive'].items()}
        n = {k: v.to(device) for k, v in batch['negative'].items()}
        va = model(a['input_ids'], a['attention_mask'])
        vp = model(p['input_ids'], p['attention_mask'])
        vn = model(n['input_ids'], n['attention_mask'])
        pos_sim = (va * vp).sum(dim=1)
        neg_sim = (va * vn).sum(dim=1)
    model.train()
    return float(pos_sim.mean().item()), float(neg_sim.mean().item())


def run_training_loop(rank, world_size, model, train_loader, train_sampler,
                      num_epochs, initial_margin, final_margin, optimizer, scheduler, dataset_id):
    device = torch.device(f'cuda:{rank}')
    criterion = TripletLoss(
        margin=initial_margin,
        mining_strategy='adaptive',
        margin_scale=3.0,
        temperature=0.12,
        regularization_weight=0.02,
        regularization_mode="proportional",
        min_reg_weight=0.0
    )

    accumulation_steps = 4
    max_grad_norm = 0.5
    is_main_process = (rank == 0)
    writer = SummaryWriter(log_dir=f"./runs/albertina_triplet_dataset_{dataset_id}") if is_main_process else None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_losses = []
        num_batches = 0
        epoch_start_time = time.time()

        cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / num_epochs))
        current_margin = final_margin + (initial_margin - final_margin) * cosine_decay
        criterion.set_margin(current_margin)

        train_sampler.set_epoch(epoch)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") if is_main_process else train_loader

        for i, batch in enumerate(progress_bar):
            try:
                anchor_inputs = {k: v.to(device) for k, v in batch['anchor'].items()}
                positive_inputs = {k: v.to(device) for k, v in batch['positive'].items()}
                negative_inputs = {k: v.to(device) for k, v in batch['negative'].items()}
                weights = batch.get('weight', None)
                if weights is not None:
                    weights = weights.to(device)

                should_sync = ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(train_loader))

                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                    anchor = model(anchor_inputs['input_ids'], anchor_inputs['attention_mask'])
                    positive = model(positive_inputs['input_ids'], positive_inputs['attention_mask'])
                    negative = model(negative_inputs['input_ids'], negative_inputs['attention_mask'])
                    loss = criterion(anchor, positive, negative, weights)
                    loss = loss / accumulation_steps

                if world_size > 1 and not should_sync:
                    with model.no_sync():
                        loss.backward()
                else:
                    loss.backward()

                if should_sync:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if not (torch.isnan(grad_norm) or grad_norm > 5.0):
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                    else:
                        logger.warning(f"[RANK {rank}] Skipping step due to unstable grad norm: {grad_norm:.4f}")
                        optimizer.zero_grad()

                batch_loss = loss.item() * accumulation_steps
                running_loss += batch_loss
                batch_losses.append(batch_loss)
                num_batches += 1

                if is_main_process and isinstance(progress_bar, tqdm):
                    lrs = scheduler.get_last_lr()
                    current_lr = lrs[0] if len(lrs) > 0 else optimizer.param_groups[0]["lr"]
                    recent_loss = sum(batch_losses[-10:]) / min(10, len(batch_losses))
                    progress_bar.set_description(
                        f"Epoch {epoch + 1}/{num_epochs}, Loss: {recent_loss:.4f}, LR: {current_lr:.2e}, Margin: {current_margin:.3f}"
                    )

            except Exception as e:
                logging.warning(f"[RANK {rank}] Batch {i} skipped due to error: {e}")
                optimizer.zero_grad()
                continue

        avg_train_loss = running_loss / max(1, num_batches)

        probe_pos, probe_neg = None, None
        try:
            probe_batch = next(iter(train_loader))
            probe_pos, probe_neg = _probe_epoch_batch_sims(model, probe_batch, device)
        except Exception:
            pass

        if writer:
            lrs = scheduler.get_last_lr()
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Learning_Rate", lrs[0] if len(lrs) > 0 else optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("Time/Epoch", time.time() - epoch_start_time, epoch)
            writer.add_scalar("Margin", current_margin, epoch)
            if probe_pos is not None:
                writer.add_scalar("Probe/PosSim", probe_pos, epoch)
            if probe_neg is not None:
                writer.add_scalar("Probe/NegSim", probe_neg, epoch)

    if writer:
        writer.close()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024,expandable_segments:True,roundup_power2_divisions:16"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_NVLS_ENABLE"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_BUFFSIZE"] = "134217728"
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    os.environ["NCCL_NET_GDR_LEVEL"] = "5"

    setup_logging()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.set_num_threads(16)
    torch.set_num_interop_threads(4)

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    logger.info(f"Running DDP: RANK {rank}, WORLD_SIZE {world_size}")
    train_distributed(rank, world_size)


if __name__ == '__main__':
    main()
