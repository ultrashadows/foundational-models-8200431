import os
import platform
import logging
import gc
import psutil
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from crawler.dgsi.helpers import connect
from encoder.dataset_processor import DatasetProcessor
from encoder.triplet_loss import TripletLoss
from encoder.wrapper import EncoderWrapper
from tqdm import tqdm


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
        logger.info("CUDA cache cleared")


def setup(rank, world_size, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    logger.info(f"[RANK {rank}] Process group initialized using {backend} backend.")


def cleanup():
    dist.destroy_process_group()
    logger.info("Cleaned up distributed process group.")


def load_latest_checkpoint(model, tokenizer, checkpoint_dir, device):
    last_epoch = -1
    if os.path.isdir(checkpoint_dir):
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("epoch_")]
        if checkpoints:
            last_epoch = max([int(d.split("_")[1]) for d in checkpoints])
            latest_checkpoint = os.path.join(checkpoint_dir, f"epoch_{last_epoch}")
            model.encoder = model.encoder.from_pretrained(latest_checkpoint).to(device)
            tokenizer = tokenizer.from_pretrained(latest_checkpoint)
    return model, tokenizer, last_epoch + 1


def save_checkpoint(model, tokenizer, epoch, checkpoint_dir):
    save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_to_save = model.module.encoder if hasattr(model, "module") else model.encoder
    model_to_save.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Checkpoint saved at {save_path}")


def train_distributed(rank, world_size, num_epochs=10):
    backend = 'nccl' if platform.system() != 'Windows' else 'gloo'
    setup(rank, world_size, backend=backend)

    collection = connect().database.encoder
    checkpoint_dir = "./checkpoints"
    log_memory_usage(rank, "[Before model loading]")

    model_name = "PORTULAN/albertina-100m-portuguese-ptpt-encoder"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EncoderWrapper(model_name=model_name).to(rank)
    model, tokenizer, start_epoch = load_latest_checkpoint(model, tokenizer, checkpoint_dir, rank)

    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
        logger.info(f"[RANK {rank}] Gradient checkpointing enabled")

    model = DDP(model, device_ids=[rank])
    logger.info(f"[RANK {rank}] Model loaded and wrapped in DDP.")
    log_memory_usage(rank, "[After model loading]")

    train_dataset = DatasetProcessor(collection=collection, tokenizer=tokenizer, split="train", max_samples=100)
    train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        sampler=train_sampler,
        pin_memory=torch.cuda.is_available(),
        num_workers=0
    )

    test_loader = None
    if rank == 0:
        test_dataset = DatasetProcessor(collection=collection, tokenizer=tokenizer, split="test", max_samples=100)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=0
        )

    run_training_loop(rank, world_size, model, tokenizer, train_loader, test_loader, train_sampler, num_epochs,
                      start_epoch, checkpoint_dir)
    cleanup()


def train_single_gpu(num_epochs=10):
    collection = connect().database.encoder
    checkpoint_dir = "./checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_memory_usage(device, "[Before model loading]")

    model_name = "PORTULAN/albertina-100m-portuguese-ptpt-encoder"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EncoderWrapper(model_name=model_name).to(device)
    model, tokenizer, start_epoch = load_latest_checkpoint(model, tokenizer, checkpoint_dir, device)

    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    logger.info("Model loaded for single-GPU training.")
    log_memory_usage(device, "[After model loading]")

    train_dataset = DatasetProcessor(collection=collection, tokenizer=tokenizer, split="train", max_samples=100)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=0
    )

    test_dataset = DatasetProcessor(collection=collection, tokenizer=tokenizer, split="test", max_samples=100)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0
    )

    run_training_loop(device, 1, model, tokenizer, train_loader, test_loader, None, num_epochs,
                      start_epoch, checkpoint_dir)


def run_training_loop(rank, world_size, model, tokenizer, train_loader, test_loader, train_sampler, num_epochs,
                      start_epoch, checkpoint_dir):
    device = rank if isinstance(rank, torch.device) else torch.device(f'cuda:{rank}' if torch.cuda.is_available()
                                                                      else 'cpu')
    model = model.to(device)
    log_memory_usage(device, "[Start of training]")

    criterion = TripletLoss(margin=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    use_fp16 = False
    scaler = GradScaler() if torch.cuda.is_available() and use_fp16 else None
    accumulation_steps = 4
    is_main_process = (rank == 0 or world_size == 1)
    writer = SummaryWriter(log_dir="./runs/albertina_triplet") if is_main_process else None

    logger.info(f"[RANK {device}] Starting training from epoch {start_epoch} to {num_epochs}...")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        if train_sampler:
            train_sampler.set_epoch(epoch)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") if is_main_process else train_loader
        optimizer.zero_grad()

        for i, batch in enumerate(progress_bar):
            anchor_inputs = {k: v.to(device) for k, v in batch['anchor'].items()}
            positive_inputs = {k: v.to(device) for k, v in batch['positive'].items()}
            negative_inputs = {k: v.to(device) for k, v in batch['negative'].items()}

            if scaler is not None:
                with autocast(device_type='cuda', dtype=torch.float16):
                    anchor_embeddings = model(anchor_inputs['input_ids'], anchor_inputs['attention_mask'])
                    positive_embeddings = model(positive_inputs['input_ids'], positive_inputs['attention_mask'])
                    negative_embeddings = model(negative_inputs['input_ids'], negative_inputs['attention_mask'])
                    loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings) / accumulation_steps
                scaler.scale(loss).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                anchor_embeddings = model(anchor_inputs['input_ids'], anchor_inputs['attention_mask'])
                positive_embeddings = model(positive_inputs['input_ids'], positive_inputs['attention_mask'])
                negative_embeddings = model(negative_inputs['input_ids'], negative_inputs['attention_mask'])
                loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings) / accumulation_steps
                loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

            if is_main_process and isinstance(progress_bar, tqdm):
                progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs}, "
                                             f"Loss: {loss.item() * accumulation_steps:.4f}")

            if (i + 1) % 20 == 0:
                log_memory_usage(device, f"[Epoch {epoch + 1}, Batch {i + 1}]")

        if scaler is not None and (len(train_loader) % accumulation_steps != 0):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        elif len(train_loader) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = running_loss / len(train_loader)
        if writer:
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        if is_main_process:
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Avg Loss: {avg_train_loss:.4f}")

            if test_loader is not None:
                model.eval()
                test_loss = 0.0
                with torch.no_grad():
                    test_progress_bar = tqdm(test_loader, desc="Evaluating") if is_main_process else test_loader
                    for batch in test_progress_bar:
                        anchor_inputs = {k: v.to(device) for k, v in batch['anchor'].items()}
                        positive_inputs = {k: v.to(device) for k, v in batch['positive'].items()}
                        negative_inputs = {k: v.to(device) for k, v in batch['negative'].items()}

                        anchor_embeddings = model(anchor_inputs['input_ids'], anchor_inputs['attention_mask'])
                        positive_embeddings = model(positive_inputs['input_ids'], positive_inputs['attention_mask'])
                        negative_embeddings = model(negative_inputs['input_ids'], negative_inputs['attention_mask'])
                        loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
                        test_loss += loss.item()

                avg_test_loss = test_loss / len(test_loader)
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Test Avg Loss: {avg_test_loss:.4f}")
                if writer:
                    writer.add_scalar("Loss/Test", avg_test_loss, epoch)

        if (epoch + 1) % 1 == 0:
            if is_main_process:
                save_checkpoint(model, tokenizer, epoch + 1, checkpoint_dir)

        empty_cache()
        log_memory_usage(device, f"[End of Epoch {epoch + 1}]")

    if is_main_process:
        log_memory_usage(device, "[Before saving model]")
        model_to_save = model.module.encoder if hasattr(model, "module") else model.encoder
        model_to_save.save_pretrained("./finetuned-albertina-triplet-loss")
        tokenizer.save_pretrained("./finetuned-albertina-triplet-loss")
        logger.info("Model and tokenizer saved successfully to ./finetuned-albertina-triplet-loss")

    if writer:
        writer.close()


def main():
    setup_logging()

    if platform.system() == 'Windows':
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("CUDNN benchmark enabled for Windows")

    world_size = torch.cuda.device_count()

    logger.info(f"Platform: {platform.system()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {world_size}")
        for i in range(world_size):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    ram = psutil.virtual_memory()
    logger.info(f"Total system RAM: {ram.total / (1024 ** 3):.2f}GB")

    if world_size > 1 and platform.system() != "Windows":
        logger.info(f"Found {world_size} GPUs. Launching DDP training...")
        mp.spawn(train_distributed, args=(world_size,), nprocs=world_size, join=True)
    else:
        logger.info("Single GPU or Windows system detected. Running in single-GPU mode.")
        train_single_gpu()


if __name__ == '__main__':
    main()
