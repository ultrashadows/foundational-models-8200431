import os
import platform
import logging
import gc
import warnings

import psutil
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from dataset_processor import DatasetProcessor
from triplet_loss import TripletLoss
from wrapper import EncoderWrapper
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel._functions")


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
            torch.cuda.set_per_process_memory_fraction(0.95, device=i)

        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,roundup_power2_divisions:16'

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
        logger.info("CUDA cache cleared and synchronized")


def setup(rank, world_size, backend='nccl'):
    master_addr = os.environ.get('MASTER_ADDR', 'gnx506')
    master_port = os.environ.get('MASTER_PORT', '12355')

    print(f"[RANK {rank}] Environment variables:")
    print(f"[RANK {rank}] MASTER_ADDR: {master_addr}")
    print(f"[RANK {rank}] MASTER_PORT: {master_port}")
    print(f"[RANK {rank}] RANK: {os.environ.get('RANK', 'not set')}")
    print(f"[RANK {rank}] LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'not set')}")
    print(f"[RANK {rank}] WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'not set')}")

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = master_addr
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = master_port

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    print(
        f"[RANK {rank}] Initialized process group with MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}")


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Cleaned up distributed process group.")


def load_latest_checkpoint(model, tokenizer, checkpoint_dir, device):
    last_epoch = -1
    if os.path.isdir(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("epoch_")]
        if checkpoints:
            last_epoch = max([int(d.split("_")[1]) for d in checkpoints])
            latest_checkpoint = os.path.join(checkpoint_dir, f"epoch_{last_epoch}")

            logger.info(f"Loading checkpoint from {latest_checkpoint}")
            model.encoder = model.encoder.from_pretrained(latest_checkpoint).to(device)
            tokenizer = tokenizer.from_pretrained(latest_checkpoint)
            logger.info(f"Resumed from epoch {last_epoch}")
    return model, tokenizer, last_epoch + 1


def save_checkpoint(model, tokenizer, epoch, checkpoint_dir, rank):
    if rank == 0:
        save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
        os.makedirs(save_path, exist_ok=True)

        model_to_save = model.module.encoder if hasattr(model, "module") else model.encoder
        model_to_save.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
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
    model, tokenizer, start_epoch = load_latest_checkpoint(model, tokenizer, checkpoint_dir, device)

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
        bucket_cap_mb=25,
        static_graph=True
    )
    logger.info(f"[RANK {rank}] Model loaded and wrapped in DDP.")
    log_memory_usage(device, "[After model loading]")

    train_batch_size = 16
    test_batch_size = 32

    train_dataset = DatasetProcessor(tokenizer=tokenizer, split="train", max_samples=10000)
    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )

    test_loader = None
    if rank == 0:
        test_dataset = DatasetProcessor(tokenizer=tokenizer, split="test", max_samples=2000)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2
        )

    run_training_loop(rank, world_size, model, tokenizer, train_loader, test_loader,
                      train_sampler, num_epochs, start_epoch, checkpoint_dir)
    cleanup()


def train_single_gpu(num_epochs=10):
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_gpu_memory()
    log_memory_usage(device, "[Before model loading]")

    model_name = "./models/albertina-1b5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = EncoderWrapper(model_name=model_name).to(device)
    model, tokenizer, start_epoch = load_latest_checkpoint(model, tokenizer, checkpoint_dir, device)

    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    logger.info("Model loaded for single-GPU training.")
    log_memory_usage(device, "[After model loading]")

    batch_size = 32

    train_dataset = DatasetProcessor(tokenizer=tokenizer, split="train", max_samples=5000)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=2
    )

    test_dataset = DatasetProcessor(tokenizer=tokenizer, split="test", max_samples=1000)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2
    )

    run_training_loop(device, 1, model, tokenizer, train_loader, test_loader,
                      None, num_epochs, start_epoch, checkpoint_dir)


def run_training_loop(rank, world_size, model, tokenizer, train_loader, test_loader,
                      train_sampler, num_epochs, start_epoch, checkpoint_dir):
    device = rank if isinstance(rank, torch.device) else torch.device(f'cuda:{rank}')
    model = model.to(device)
    log_memory_usage(device, "[Start of training]")

    criterion = TripletLoss(margin=1.0)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    total_steps = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-5,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )

    use_fp16 = False
    scaler = GradScaler() if use_fp16 else None

    accumulation_steps = 2

    is_main_process = (rank == 0 or world_size == 1)
    writer = SummaryWriter(log_dir="./runs/albertina_triplet") if is_main_process else None

    if hasattr(model, 'module') and world_size > 1:
        model._set_static_graph()
        logger.info(f"[RANK {device}] Static graph set for DDP")

    logger.info(f"[RANK {device}] Starting training with FP16={use_fp16}, accumulation_steps={accumulation_steps}")
    logger.info(f"[RANK {device}] Batch size per GPU: {train_loader.batch_size}")
    logger.info(f"[RANK {device}] Effective batch size: {train_loader.batch_size * world_size * accumulation_steps}")
    logger.info(f"[RANK {device}] Starting training from epoch {start_epoch} to {num_epochs}...")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        if train_sampler:
            train_sampler.set_epoch(epoch)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") if is_main_process else train_loader
        optimizer.zero_grad()

        for i, batch in enumerate(progress_bar):
            anchor_inputs = {k: v.to(device, non_blocking=True) for k, v in batch['anchor'].items()}
            positive_inputs = {k: v.to(device, non_blocking=True) for k, v in batch['positive'].items()}
            negative_inputs = {k: v.to(device, non_blocking=True) for k, v in batch['negative'].items()}

            should_sync = (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader)

            if use_fp16 and scaler is not None:
                with autocast(device_type='cuda', dtype=torch.float16):
                    anchor_embeddings = model(anchor_inputs['input_ids'], anchor_inputs['attention_mask'])
                    positive_embeddings = model(positive_inputs['input_ids'], positive_inputs['attention_mask'])
                    negative_embeddings = model(negative_inputs['input_ids'], negative_inputs['attention_mask'])
                    loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings) / accumulation_steps

                if hasattr(model, 'no_sync') and world_size > 1 and not should_sync:
                    with model.no_sync():
                        scaler.scale(loss).backward()
                else:
                    scaler.scale(loss).backward()

                if should_sync:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                anchor_embeddings = model(anchor_inputs['input_ids'], anchor_inputs['attention_mask'])
                positive_embeddings = model(positive_inputs['input_ids'], positive_inputs['attention_mask'])
                negative_embeddings = model(negative_inputs['input_ids'], negative_inputs['attention_mask'])
                loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings) / accumulation_steps

                if hasattr(model, 'no_sync') and world_size > 1 and not should_sync:
                    with model.no_sync():
                        loss.backward()
                else:
                    loss.backward()

                if should_sync:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            num_batches += 1

            if is_main_process and isinstance(progress_bar, tqdm):
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_description(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f"Loss: {loss.item() * accumulation_steps:.4f}, "
                    f"LR: {current_lr:.2e}"
                )

            if (i + 1) % 50 == 0:
                empty_cache()
                if (i + 1) % 100 == 0:
                    log_memory_usage(device, f"[Epoch {epoch + 1}, Batch {i + 1}]")

        if len(train_loader) % accumulation_steps != 0:
            if use_fp16 and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = running_loss / num_batches if num_batches > 0 else 0

        if writer:
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch)

        if is_main_process:
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Avg Loss: {avg_train_loss:.4f}")

            if test_loader is not None:
                model.eval()
                test_loss = 0.0
                test_batches = 0

                with torch.no_grad():
                    test_progress_bar = tqdm(test_loader, desc="Evaluating") if is_main_process else test_loader
                    for batch in test_progress_bar:
                        anchor_inputs = {k: v.to(device, non_blocking=True) for k, v in batch['anchor'].items()}
                        positive_inputs = {k: v.to(device, non_blocking=True) for k, v in batch['positive'].items()}
                        negative_inputs = {k: v.to(device, non_blocking=True) for k, v in batch['negative'].items()}

                        if use_fp16:
                            with autocast(device_type='cuda', dtype=torch.float16):
                                anchor_embeddings = model(anchor_inputs['input_ids'], anchor_inputs['attention_mask'])
                                positive_embeddings = model(positive_inputs['input_ids'],
                                                            positive_inputs['attention_mask'])
                                negative_embeddings = model(negative_inputs['input_ids'],
                                                            negative_inputs['attention_mask'])
                                loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
                        else:
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

        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, tokenizer, epoch + 1, checkpoint_dir, rank)

        empty_cache()
        log_memory_usage(device, f"[End of Epoch {epoch + 1}]")

    if is_main_process:
        log_memory_usage(device, "[Before saving model]")
        model_to_save = model.module.encoder if hasattr(model, "module") else model.encoder
        final_save_path = "./finetuned-albertina-1b5-triplet-loss"
        model_to_save.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        logger.info(f"Model and tokenizer saved successfully to {final_save_path}")

    if writer:
        writer.close()


def main():
    setup_logging()

    if platform.system() != 'Windows':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("CUDNN optimizations enabled")

    world_size = torch.cuda.device_count()

    logger.info(f"Platform: {platform.system()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {world_size}")
        for i in range(world_size):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} - "
                        f"Compute Capability: {props.major}.{props.minor}, "
                        f"Memory: {props.total_memory / (1024 ** 3):.1f}GB")

    ram = psutil.virtual_memory()
    logger.info(f"Total system RAM: {ram.total / (1024 ** 3):.2f}GB")

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size_env = int(os.environ.get("WORLD_SIZE", 1))

    if world_size_env > 1:
        logger.info(f"Running DDP: RANK {rank}, LOCAL_RANK {local_rank}, WORLD_SIZE {world_size_env}")
        train_distributed(local_rank, world_size_env)
    else:
        logger.info("Single GPU or environment not set up for DDP. Running in single-GPU mode.")
        train_single_gpu()


if __name__ == '__main__':
    main()
