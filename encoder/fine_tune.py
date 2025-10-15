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
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from dataset_processor import DatasetProcessor
from triplet_loss import TripletLoss
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel._functions")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)
DEFAULT_MARGIN = float(os.environ.get("TRIPLET_MARGIN", "0.2066"))


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
                torch.cuda.set_per_process_memory_fraction(0.98, device=i)
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


def choose_model_load_path(dataset_id: int) -> str:
    if dataset_id == 1:
        return "./models/albertina-1b5"
    prev_path = f"./results/finetuned-albertina-1b5-triplet-loss-dataset-{dataset_id - 1}"
    return prev_path if os.path.isdir(prev_path) else "./models/albertina-1b5"


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


def train_distributed(rank, world_size, num_epochs=2, start_index=1, num_datasets=1):
    torch.cuda.set_device(rank)
    timeout = timedelta(minutes=int(os.environ.get("DDP_TIMEOUT_MIN", "60")))
    dist.init_process_group(backend="nccl", init_method="env://", timeout=timeout)
    dist.barrier()
    if _ddp_health_check(rank) is None:
        raise RuntimeError("DDP health-check failed right after init; check NCCL env and network interfaces.")

    device = torch.device(f"cuda:{rank}")
    configure_gpu_memory()

    for dataset_id in range(start_index, start_index + num_datasets):
        dataset_name = f"dataset_{dataset_id}.json"
        load_path = choose_model_load_path(dataset_id)

        if rank == 0:
            logger.info(f"Starting training on {dataset_name} using model: {load_path} (margin={DEFAULT_MARGIN:.4f})")

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
            [{"params": model.parameters(), "weight_decay": 0.01, "lr": 2e-5}],
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        train_dataset = DatasetProcessor(tokenizer, split="train", max_length=512, dataset=dataset_name,
                                         hard_negatives=False)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
                                           drop_last=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=8,
            persistent_workers=False,
            prefetch_factor=2,
            drop_last=True,
        )

        accumulation_steps = 8
        opt_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
        total_opt_steps = opt_steps_per_epoch * num_epochs
        warmup_steps = max(50, int(0.08 * total_opt_steps))

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_opt_steps
        )

        run_training_loop(
            rank, world_size, model, train_loader, train_sampler,
            num_epochs, optimizer, scheduler, dataset_id
        )

        if rank == 0:
            save_dir = f"./results/finetuned-albertina-1b5-triplet-loss-dataset-{dataset_id}"
            save_encoder(model.module if hasattr(model, "module") else model, tokenizer, save_dir)

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
                      num_epochs, optimizer, scheduler, dataset_id):
    device = torch.device(f"cuda:{rank}")
    criterion = TripletLoss(margin=DEFAULT_MARGIN)
    accumulation_steps = 8
    max_grad_norm = 0.5
    is_main_process = (rank == 0)
    writer = SummaryWriter(log_dir=f"./runs/albertina_triplet_dataset_{dataset_id}") if is_main_process else None

    for epoch in range(num_epochs):
        model.train()
        running_total = 0.0
        num_batches = 0
        epoch_start_time = time.time()

        train_sampler.set_epoch(epoch)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") if is_main_process else train_loader

        for i, batch in enumerate(progress_bar):
            try:
                a = {k: v.to(device) for k, v in batch["anchor"].items()}
                p = {k: v.to(device) for k, v in batch["positive"].items()}
                n = {k: v.to(device) for k, v in batch["negative"].items()}

                should_sync = ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(train_loader))

                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    a_emb, p_emb, n_emb = single_forward_embed(
                        model, a["input_ids"], a["attention_mask"],
                        p["input_ids"], p["attention_mask"],
                        n["input_ids"], n["attention_mask"]
                    )
                    loss_val = criterion(a_emb, p_emb, n_emb)
                    loss = loss_val / accumulation_steps

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
                        logging.warning(f"[RANK {rank}] Skipping step due to unstable grad norm: {grad_norm:.4f}")
                        optimizer.zero_grad()

                running_total += float(loss_val.item())
                num_batches += 1

                if is_main_process and isinstance(progress_bar, tqdm):
                    lrs = scheduler.get_last_lr()
                    current_lr = lrs[0] if len(lrs) > 0 else optimizer.param_groups[0]["lr"]
                    recent_total = running_total / max(1, num_batches)
                    progress_bar.set_description(
                        f"Epoch {epoch + 1}/{num_epochs}, Loss: {recent_total:.4f}, LR: {current_lr:.2e}, Margin: {DEFAULT_MARGIN:.3f}"
                    )
            except Exception as e:
                logging.error(f"[RANK {rank}] Fatal batch error at step {i}: {e}")
                optimizer.zero_grad()
                if dist.is_initialized():
                    dist.barrier()
                raise

        avg_total = running_total / max(1, num_batches)

        probe_pos, probe_neg = None, None
        try:
            probe_batch = next(iter(train_loader))
            probe_pos, probe_neg = _probe_epoch_batch_sims(model, probe_batch, device)
        except Exception:
            pass

        if writer:
            lrs = scheduler.get_last_lr()
            writer.add_scalar("Loss/Train_total", avg_total, epoch)
            writer.add_scalar("Learning_Rate", lrs[0] if len(lrs) > 0 else optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("Time/Epoch", time.time() - epoch_start_time, epoch)
            writer.add_scalar("Margin", DEFAULT_MARGIN, epoch)
            if probe_pos is not None:
                writer.add_scalar("Probe/PosSim", probe_pos, epoch)
            if probe_neg is not None:
                writer.add_scalar("Probe/NegSim", probe_neg, epoch)

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
                          "max_split_size_mb:1024,expandable_segments:True,roundup_power2_divisions:16")

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
