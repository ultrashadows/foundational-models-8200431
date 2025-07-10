import os
import platform
import logging
import gc
import warnings
import time
import multiprocessing as mp
from datetime import timedelta

import psutil
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from dataset_processor import DatasetProcessor
from triplet_loss import TripletLoss
from wrapper import EncoderWrapper

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel._functions")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_logging(log_file="training.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


logger = logging.getLogger(__name__)


def configure_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(0.98, device=i)

        os.environ[
            'PYTORCH_CUDA_ALLOC_CONF'] = ('max_split_size_mb:128,expandable_segments:True,roundup_power2_divisions:8,'
                                          'garbage_collection_threshold:0.8')

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
        f"{prefix} System RAM: {ram.used / (1024 ** 3):.2f}GB used, "
        f"{ram.available / (1024 ** 3):.2f}GB available ({ram.percent}%)")


def empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def setup(rank, world_size, backend='nccl'):
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12355')

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = master_addr
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = master_port

    os.environ['NCCL_ALGO'] = 'Ring'
    os.environ['NCCL_PROTO'] = 'Simple'
    os.environ['NCCL_MIN_NRINGS'] = '16'
    os.environ['NCCL_MAX_NRINGS'] = '32'
    os.environ['NCCL_BUFFSIZE'] = '67108864'
    os.environ['NCCL_NTHREADS'] = '2048'
    os.environ['NCCL_TREE_THRESHOLD'] = '8589934592'
    os.environ['NCCL_IB_DISABLE'] = '0'
    os.environ['NCCL_IB_HCA'] = '^mlx5'
    os.environ['NCCL_IB_GID_INDEX'] = '3'
    os.environ['NCCL_NET_GDR_LEVEL'] = '5'
    os.environ['NCCL_P2P_LEVEL'] = 'NVL'
    os.environ['NCCL_NSOCKS_PERTHREAD'] = '8'
    os.environ['NCCL_SOCKET_NTHREADS'] = '8'
    os.environ['NCCL_REDUCE_ALGO'] = 'Ring'
    os.environ['NCCL_ALLREDUCE_ALGO'] = 'Ring'

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=30)
    )
    logger.info(f"[RANK {rank}] Process group initialized")


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_latest_checkpoint(model, optimizer, scheduler, scaler, checkpoint_dir, device):
    last_epoch = -1
    if os.path.isdir(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("epoch_")]
        if checkpoints:
            last_epoch = max([int(d.split("_")[1]) for d in checkpoints])
            latest_checkpoint = os.path.join(checkpoint_dir, f"epoch_{last_epoch}")

            logger.info(f"Loading checkpoint from {latest_checkpoint}")

            if hasattr(model, 'module'):
                model.module.encoder = model.module.encoder.from_pretrained(latest_checkpoint).to(device)
            else:
                model.encoder = model.encoder.from_pretrained(latest_checkpoint).to(device)

            state_file = os.path.join(latest_checkpoint, "training_state.pt")
            if os.path.exists(state_file):
                state = torch.load(state_file, map_location=device)
                optimizer.load_state_dict(state['optimizer'])
                scheduler.load_state_dict(state['scheduler'])
                if scaler and 'scaler' in state:
                    scaler.load_state_dict(state['scaler'])
                logger.info(f"Loaded optimizer and scheduler state from epoch {last_epoch}")

            logger.info(f"Resumed from epoch {last_epoch}")
    return last_epoch + 1


def save_checkpoint(model, optimizer, scheduler, scaler, tokenizer, epoch, checkpoint_dir, rank):
    if rank == 0:
        save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
        os.makedirs(save_path, exist_ok=True)

        model_to_save = model.module.encoder if hasattr(model, "module") else model.encoder
        model_to_save.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        state = {
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }
        if scaler:
            state['scaler'] = scaler.state_dict()

        torch.save(state, os.path.join(save_path, "training_state.pt"))
        logger.info(f"Checkpoint saved at {save_path}")

    if dist.is_initialized():
        dist.barrier()


def train_distributed(rank, world_size, num_epochs=10):
    backend = 'nccl' if platform.system() != 'Windows' else 'gloo'
    setup(rank, world_size, backend=backend)

    configure_gpu_memory()

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_memory_usage(device, "[Before model loading]")

    model_name = "./models/albertina-1b5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = EncoderWrapper(model_name=model_name).to(device)

    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
        logger.info(f"[RANK {rank}] Gradient checkpointing enabled")

    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        bucket_cap_mb=512,
        static_graph=True,
    )
    logger.info(f"[RANK {rank}] Model loaded and wrapped in DDP.")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
        foreach=True
    )

    train_batch_size = 512
    test_batch_size = 512

    max_length = 512

    train_dataset = DatasetProcessor(
        tokenizer=tokenizer,
        split="train",
        max_length=max_length,
        max_samples=10000,
    )

    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    num_workers = min(32, mp.cpu_count())
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=16,
        drop_last=True,
        multiprocessing_context='spawn'
    )

    test_loader = None
    if rank == 0:
        test_dataset = DatasetProcessor(
            tokenizer=tokenizer,
            split="test",
            max_length=max_length,
            max_samples=10000
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers // 2,
            persistent_workers=True,
            prefetch_factor=16,
            drop_last=False,
        )

    total_steps = len(train_loader) * num_epochs

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    use_bf16 = True
    scaler = None

    start_epoch = load_latest_checkpoint(model, optimizer, scheduler, scaler, checkpoint_dir, device)

    run_training_loop(rank, world_size, model, tokenizer, train_dataset, train_loader, test_loader,
                      train_sampler, num_epochs, start_epoch, checkpoint_dir,
                      optimizer, scheduler, scaler, use_bf16)
    cleanup()


def run_training_loop(rank, world_size, model, tokenizer, train_dataset, train_loader, test_loader,
                      train_sampler, num_epochs, start_epoch, checkpoint_dir,
                      optimizer, scheduler, scaler, use_bf16):
    device = rank if isinstance(rank, torch.device) else torch.device(f'cuda:{rank}')
    model = model.to(device)

    criterion = TripletLoss(margin=0.5)

    is_main_process = (rank == 0 or world_size == 1)
    writer = SummaryWriter(log_dir="./runs/albertina_triplet") if is_main_process else None

    effective_batch_size = train_loader.batch_size * world_size
    logger.info(f"[RANK {rank}] Starting training with BF16={use_bf16}")
    logger.info(f"[RANK {rank}] Effective batch size: {effective_batch_size}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()

        if train_sampler:
            train_sampler.set_epoch(epoch)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") if is_main_process else train_loader

        batch_iterator = train_loader
        if is_main_process:
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

        for i, batch in enumerate(batch_iterator):
            anchor_inputs = {k: v.to(device, non_blocking=True).contiguous() for k, v in batch['anchor'].items()}
            positive_inputs = {k: v.to(device, non_blocking=True).contiguous() for k, v in batch['positive'].items()}
            negative_inputs = {k: v.to(device, non_blocking=True).contiguous() for k, v in batch['negative'].items()}

            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
                anchor_embeddings = model(anchor_inputs['input_ids'], anchor_inputs['attention_mask'])
                positive_embeddings = model(positive_inputs['input_ids'], positive_inputs['attention_mask'])
                negative_embeddings = model(negative_inputs['input_ids'], negative_inputs['attention_mask'])
                loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            num_batches += 1

            if is_main_process:
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_description(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f"Loss: {loss.item()}, "
                    f"LR: {current_lr:.2e}"
                )
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}, Batch {i}, "
                    f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}"
                )

        epoch_time = time.time() - epoch_start_time
        avg_train_loss = running_loss / num_batches if num_batches > 0 else 0

        if writer:
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("Time/Epoch", epoch_time, epoch)

        if is_main_process:
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}, Train Avg Loss: {avg_train_loss:.4f}, "
                f"Time: {epoch_time:.2f}s, Samples/sec: {len(train_dataset) / epoch_time:.1f}")

            if test_loader is not None and (epoch + 1) % 5 == 0:
                model.eval()
                test_loss = 0.0
                test_batches = 0

                with torch.no_grad():
                    for batch in test_loader:
                        anchor_inputs = {k: v.to(device, non_blocking=True).contiguous() for k, v in
                                         batch['anchor'].items()}
                        positive_inputs = {k: v.to(device, non_blocking=True).contiguous() for k, v in
                                           batch['positive'].items()}
                        negative_inputs = {k: v.to(device, non_blocking=True).contiguous() for k, v in
                                           batch['negative'].items()}

                        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
                            anchor_embeddings = model(anchor_inputs['input_ids'], anchor_inputs['attention_mask'])
                            positive_embeddings = model(positive_inputs['input_ids'], positive_inputs['attention_mask'])
                            negative_embeddings = model(negative_inputs['input_ids'], negative_inputs['attention_mask'])
                            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

                        test_loss += loss.item()
                        test_batches += 1

                avg_test_loss = test_loss / test_batches if test_batches > 0 else 0
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Test Avg Loss: {avg_test_loss:.4f}")
                if writer:
                    writer.add_scalar("Loss/Test", avg_test_loss, epoch)

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, tokenizer, epoch + 1, checkpoint_dir, rank)

        if (epoch + 1) % 5 == 0:
            empty_cache()

    if is_main_process:
        model_to_save = model.module.encoder if hasattr(model, "module") else model.encoder
        final_save_path = "./finetuned-albertina-1b5-triplet-loss"
        model_to_save.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        logger.info(f"Model and tokenizer saved to {final_save_path}")

    if writer:
        writer.close()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    setup_logging()

    if platform.system() != 'Windows':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cudnn.enabled = True

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        torch.set_num_threads(min(64, os.cpu_count()))
        torch.set_num_interop_threads(min(32, os.cpu_count() // 2))

        logger.info("All CUDA optimizations enabled")

    world_size = torch.cuda.device_count()
    logger.info(f"Available GPUs: {world_size}")

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size_env = int(os.environ.get("WORLD_SIZE", 1))

    if world_size_env > 1:
        logger.info(f"Running DDP: RANK {rank}, LOCAL_RANK {local_rank}, WORLD_SIZE {world_size_env}")
        train_distributed(local_rank, world_size_env)
    else:
        logger.info("Running in single-GPU mode")
        train_distributed(0, 1)


if __name__ == '__main__':
    main()