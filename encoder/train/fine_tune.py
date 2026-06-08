import math
import os
import logging
import gc
import warnings
import time
from datetime import timedelta
import psutil
import torch
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from dataset_processor import DatasetProcessor
from triplet_loss import TripletLoss
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel._functions")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)
DEFAULT_MARGIN = 0.06


def setup_logging(log_file="training.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )


def configure_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_per_process_memory_fraction(0.95, device=i)
            except Exception:
                pass
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} - Total Memory: {total_memory:.2f}GB")


def log_memory_usage(device, prefix=""):
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        logger.info(f"{prefix} GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
    ram = psutil.virtual_memory()
    logger.info(
        f"{prefix} System RAM: {ram.used / (1024 ** 3):.2f}GB used, {ram.available / (1024 ** 3):.2f}GB available ({ram.percent}%)")


def empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def save_encoder(model, tokenizer, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Model and tokenizer saved to {save_dir}")


def _ddp_health_check(rank):
    try:
        t = torch.tensor([1.0], device=torch.device(f"cuda:{rank}"))
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t.item())
    except Exception as e:
        logger.error(f"[RANK {rank}] DDP health-check failed: {e}")
        return None


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def single_forward_embed(model, a_ids, a_mask, p_ids, p_mask, n_ids, n_mask):
    input_ids = torch.cat([a_ids, p_ids, n_ids], dim=0)
    attn_mask = torch.cat([a_mask, p_mask, n_mask], dim=0)
    out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=False, return_dict=True)
    pooled = mean_pool(out.last_hidden_state, attn_mask)
    normed = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    b = a_ids.size(0)
    return normed[:b], normed[b:2 * b], normed[2 * b:]


def calculate_data_margin(train_loader, model, device, sample_size=1000):
    model.eval()
    margins = []

    logger.info("Calculating data margin statistics...")
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= sample_size // train_loader.batch_size:
                break

            a = {k: v.to(device) for k, v in batch["anchor"].items()}
            p = {k: v.to(device) for k, v in batch["positive"].items()}
            n = {k: v.to(device) for k, v in batch["negative"].items()}

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                a_emb, p_emb, n_emb = single_forward_embed(
                    model, a["input_ids"], a["attention_mask"],
                    p["input_ids"], p["attention_mask"],
                    n["input_ids"], n["attention_mask"]
                )
                pos_sim = (a_emb * p_emb).sum(dim=1)
                neg_sim = (a_emb * n_emb).sum(dim=1)
                batch_margins = pos_sim - neg_sim
                margins.extend(batch_margins.cpu().tolist())

    model.train()
    margins = torch.tensor(margins)

    mean_margin = margins.mean().item()
    median_margin = margins.median().item()
    std_margin = margins.std().item()
    p25_margin = margins.quantile(0.25).item()
    p75_margin = margins.quantile(0.75).item()
    p10_margin = margins.quantile(0.10).item()

    suggested_margin = max(0.03, min(p25_margin * 0.7, median_margin * 0.5))

    stats = {
        'mean': mean_margin,
        'median': median_margin,
        'std': std_margin,
        'p10': p10_margin,
        'p25': p25_margin,
        'p75': p75_margin,
        'min': margins.min().item(),
        'max': margins.max().item()
    }

    return suggested_margin, stats


def train_distributed(rank, world_size, num_epochs=2):
    torch.cuda.set_device(rank)
    timeout = timedelta(minutes=int(os.environ.get("DDP_TIMEOUT_MIN", "60")))
    dist.init_process_group(backend="nccl", init_method="env://", timeout=timeout)
    dist.barrier()
    if _ddp_health_check(rank) is None:
        raise RuntimeError("DDP health-check failed right after init; check NCCL env and network interfaces.")

    device = torch.device(f"cuda:{rank}")
    configure_gpu_memory()

    dataset_name = "dataset_1.json"
    load_path = "./models/finetuned-albertina-1b5-100k-triplets"

    if rank == 0:
        logger.info(f"Starting training on {dataset_name} for {num_epochs} epochs")
        logger.info(f"Using model: {load_path}")

    tokenizer = AutoTokenizer.from_pretrained(load_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.cls_token or tokenizer.unk_token

    base_model = AutoModel.from_pretrained(load_path)
    try:
        base_model.gradient_checkpointing_enable()
    except Exception:
        pass
    base_model.to(device)

    model = DDP(
        base_model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        bucket_cap_mb=128,
        static_graph=True
    )

    optimizer = torch.optim.AdamW(
        [{"params": model.parameters(), "weight_decay": 0.01, "lr": 2e-6}],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    train_dataset = DatasetProcessor(
        tokenizer,
        split="train",
        max_length=512,
        dataset=dataset_name,
        hard_negatives=False
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=12,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

    accumulation_steps = 6
    opt_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_opt_steps = opt_steps_per_epoch * num_epochs
    warmup_steps = max(200, int(0.10 * total_opt_steps))

    if rank == 0:
        suggested_margin, margin_stats = calculate_data_margin(train_loader, model, device)
        logger.info(f"Data margin statistics:")
        logger.info(f"  Mean: {margin_stats['mean']:.4f}, Median: {margin_stats['median']:.4f}")
        logger.info(f"  Std: {margin_stats['std']:.4f}")
        logger.info(f"  P10: {margin_stats['p10']:.4f}, P25: {margin_stats['p25']:.4f}, P75: {margin_stats['p75']:.4f}")
        logger.info(f"  Min: {margin_stats['min']:.4f}, Max: {margin_stats['max']:.4f}")
        logger.info(f"  Suggested base margin: {suggested_margin:.4f}")

        margin_tensor = torch.tensor([suggested_margin], device=device)
    else:
        margin_tensor = torch.zeros(1, device=device)

    dist.broadcast(margin_tensor, src=0)
    adaptive_margin = margin_tensor.item()

    if rank == 0:
        logger.info(f"Training configuration:")
        logger.info(f"  Dataset size: {len(train_dataset):,} triplets")
        logger.info(f"  Batch size: 12 x {accumulation_steps} accumulation = effective 72")
        logger.info(f"  Steps per epoch: {opt_steps_per_epoch:,}")
        logger.info(f"  Total steps: {total_opt_steps:,}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Learning rate: 5e-6")
        logger.info(f"  Using AdaptiveTripletLoss with base margin: {adaptive_margin:.4f}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps
    )

    scaler = GradScaler()

    run_training_loop(
        rank, world_size, model, train_loader, train_sampler,
        num_epochs, optimizer, scheduler, accumulation_steps, tokenizer, scaler, adaptive_margin
    )

    if rank == 0:
        save_dir = "./results/finetuned-albertina-1b5-100k-triplets"
        save_encoder(model.module if hasattr(model, "module") else model, tokenizer, save_dir)
        logger.info(f"Training complete! Model saved to {save_dir}")

    dist.barrier()
    torch.cuda.empty_cache()
    gc.collect()

    dist.destroy_process_group()


def _probe_epoch_batch_sims(model, batch, device):
    model.eval()
    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        a = {k: v.to(device) for k, v in batch["anchor"].items()}
        p = {k: v.to(device) for k, v in batch["positive"].items()}
        n = {k: v.to(device) for k, v in batch["negative"].items()}
        a_emb, p_emb, n_emb = single_forward_embed(model, a["input_ids"], a["attention_mask"],
                                                   p["input_ids"], p["attention_mask"],
                                                   n["input_ids"], n["attention_mask"])
        pos_sim = (a_emb * p_emb).sum(dim=1)
        neg_sim = (a_emb * n_emb).sum(dim=1)
    model.train()
    return float(pos_sim.mean().item()), float(neg_sim.mean().item())


def run_training_loop(rank, world_size, model, train_loader, train_sampler,
                      num_epochs, optimizer, scheduler, accumulation_steps, tokenizer, scaler, base_margin):
    device = torch.device(f"cuda:{rank}")
    criterion = TripletLoss(base_margin=base_margin)
    max_grad_norm = 1.0
    is_main_process = (rank == 0)
    writer = SummaryWriter(log_dir=f"./runs/albertina_100k_triplets") if is_main_process else None

    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 2

    skip_count = 0
    total_steps = 0

    for epoch in range(num_epochs):
        model.train()
        running_total = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        epoch_skip_count = 0

        train_sampler.set_epoch(epoch)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") if is_main_process else train_loader

        for i, batch in enumerate(progress_bar):
            try:
                a = {k: v.to(device) for k, v in batch["anchor"].items()}
                p = {k: v.to(device) for k, v in batch["positive"].items()}
                n = {k: v.to(device) for k, v in batch["negative"].items()}
                w = batch["weight"].to(device)

                should_sync = ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(train_loader))

                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    a_emb, p_emb, n_emb = single_forward_embed(
                        model, a["input_ids"], a["attention_mask"],
                        p["input_ids"], p["attention_mask"],
                        n["input_ids"], n["attention_mask"]
                    )
                    loss_val = criterion(a_emb, p_emb, n_emb, sample_weights=w)

                    if torch.isnan(loss_val) or torch.isinf(loss_val):
                        if is_main_process:
                            logger.warning(f"[RANK {rank}] Invalid loss detected, skipping batch")
                        optimizer.zero_grad()
                        continue

                    loss = loss_val / accumulation_steps

                scaler.scale(loss).backward()

                if should_sync:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        if is_main_process:
                            logger.warning(f"[RANK {rank}] Skipping step due to NaN/Inf grad norm")
                        scaler.update()
                        optimizer.zero_grad()
                        epoch_skip_count += 1
                        skip_count += 1
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                        total_steps += 1

                running_total += float(loss_val.item())
                num_batches += 1

                if is_main_process and isinstance(progress_bar, tqdm):
                    lrs = scheduler.get_last_lr()
                    current_lr = lrs[0] if len(lrs) > 0 else optimizer.param_groups[0]["lr"]
                    recent_total = running_total / max(1, num_batches)
                    progress_bar.set_description(
                        f"Epoch {epoch + 1}/{num_epochs}, Loss: {recent_total:.4f}, LR: {current_lr:.2e}, Skips: {epoch_skip_count}"
                    )
            except Exception as e:
                logging.error(f"[RANK {rank}] Fatal batch error at step {i}: {e}")
                optimizer.zero_grad()
                if dist.is_initialized():
                    dist.barrier()
                raise

        avg_total = running_total / max(1, num_batches)
        epoch_time = time.time() - epoch_start_time

        probe_pos, probe_neg = None, None
        try:
            probe_batch = next(iter(train_loader))
            probe_pos, probe_neg = _probe_epoch_batch_sims(model, probe_batch, device)
        except Exception:
            pass

        if is_main_process:
            logger.info(f"Epoch {epoch + 1}/{num_epochs} Summary:")
            logger.info(f"  Loss: {avg_total:.4f}")
            logger.info(f"  Time: {epoch_time:.1f}s")
            logger.info(f"  Skipped steps: {epoch_skip_count} ({skip_count} total)")
            if probe_pos and probe_neg:
                logger.info(
                    f"  Probe similarities - Pos: {probe_pos:.3f}, Neg: {probe_neg:.3f}, Margin: {probe_pos - probe_neg:.3f}")

        if writer:
            lrs = scheduler.get_last_lr()
            writer.add_scalar("Loss/Train_total", avg_total, epoch)
            writer.add_scalar("Learning_Rate", lrs[0] if len(lrs) > 0 else optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("Time/Epoch", epoch_time, epoch)
            writer.add_scalar("Base_Margin", base_margin, epoch)
            writer.add_scalar("Skipped_Steps", epoch_skip_count, epoch)
            if probe_pos is not None:
                writer.add_scalar("Probe/PosSim", probe_pos, epoch)
                writer.add_scalar("Probe/NegSim", probe_neg, epoch)
                writer.add_scalar("Probe/Margin", probe_pos - probe_neg, epoch)

        if avg_total < best_loss:
            best_loss = avg_total
            patience_counter = 0
            if is_main_process:
                logger.info(f"  New best loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if is_main_process:
                logger.info(f"  No improvement for {patience_counter} epoch(s)")

        if is_main_process and epoch < num_epochs - 1:
            checkpoint_dir = f"./results/checkpoint_epoch_{epoch + 1}"
            save_encoder(model.module if hasattr(model, "module") else model, tokenizer, checkpoint_dir)
            logger.info(f"  Checkpoint saved to {checkpoint_dir}")

        if patience_counter >= patience_limit and epoch >= 1:
            if is_main_process:
                logger.info(f"Early stopping triggered after epoch {epoch + 1}")
            break

    if writer:
        writer.close()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "max_split_size_mb:512,expandable_segments:True,roundup_power2_divisions:16")

    setup_logging()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cudnn.enabled = True
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except Exception:
        pass

    try:
        torch.set_num_threads(max(1, os.cpu_count() // 2))
        torch.set_num_interop_threads(2)
    except Exception:
        pass

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    logger.info(f"Running DDP: RANK {rank}, WORLD_SIZE {world_size}")
    train_distributed(rank, world_size)


if __name__ == "__main__":
    main()