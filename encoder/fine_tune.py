import os
import logging
import gc
import warnings
import time
import psutil
import torch
import torch.distributed as dist
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from dataset_processor import DatasetProcessor
from triplet_loss import TripletLoss
from wrapper import EncoderWrapper
from tqdm import tqdm

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


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None, -1

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if
                        f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
    if not checkpoint_files:
        return None, -1

    epochs = [int(f.split("_")[2].split(".")[0]) for f in checkpoint_files]
    last_epoch = max(epochs)
    latest_checkpoint = os.path.join(checkpoint_dir, f"checkpoint_epoch_{last_epoch}.pt")

    return latest_checkpoint, last_epoch


def load_model_from_checkpoint_or_base(base_model_path, checkpoint_dir, device, rank):
    latest_checkpoint, last_epoch = find_latest_checkpoint(checkpoint_dir)

    model = EncoderWrapper(model_name=base_model_path).to(device)
    start_epoch = 0

    if latest_checkpoint and os.path.exists(latest_checkpoint):
        if rank == 0:
            logger.info(f"Loading checkpoint from: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            logger.info(f"Loaded checkpoint from epoch {start_epoch}")

        if dist.is_initialized():
            start_epoch_list = [start_epoch]
            dist.broadcast_object_list(start_epoch_list, src=0)
            start_epoch = start_epoch_list[0]
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
    else:
        logger.info(f"No checkpoint found. Starting from base model: {base_model_path}")
        if dist.is_initialized():
            for param in model.parameters():
                dist.broadcast(param.data, src=0)

    return model, start_epoch


def save_checkpoint(model, epoch, checkpoint_dir, rank):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'epoch': epoch
        }
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        cleanup_old_checkpoints(checkpoint_dir, keep_last=2)


def cleanup_old_checkpoints(checkpoint_dir, keep_last=2):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if
                        f.startswith("checkpoint_epoch_") and f.endswith(".pt")]

    if len(checkpoint_files) > keep_last:
        checkpoint_files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
        for old_checkpoint in checkpoint_files[:-keep_last]:
            old_path = os.path.join(checkpoint_dir, old_checkpoint)
            try:
                os.remove(old_path)
                logger.info(f"Removed old checkpoint: {old_path}")
            except OSError as e:
                logger.warning(f"Could not remove old checkpoint {old_path}: {e}")


def train_distributed(rank, world_size, num_epochs=10):
    dist.init_process_group(backend='nccl')

    configure_gpu_memory()
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    checkpoint_dir = "./checkpoints"
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    base_model_path = "./models/albertina-1b5"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, start_epoch = load_model_from_checkpoint_or_base(base_model_path, checkpoint_dir, device, rank)

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

    optimizer = optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True
    )

    train_batch_size = 32
    max_length = 512

    train_dataset = DatasetProcessor(
        tokenizer=tokenizer,
        split="train",
        max_length=max_length,
        dataset="dataset_1.json"
    )

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
        num_workers=32,
        persistent_workers=True,
        prefetch_factor=8,
        drop_last=True,
    )

    test_loader = None
    if rank == 0:
        test_dataset = DatasetProcessor(
            tokenizer=tokenizer,
            split="test",
            max_length=max_length,
            dataset="dataset_1.json"
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=False,
        )

    total_steps = len(train_loader) * num_epochs

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=4e-4,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=5,
    )

    run_training_loop(rank, world_size, model, tokenizer, train_dataset, train_loader, test_loader,
                      train_sampler, num_epochs, start_epoch, checkpoint_dir,
                      optimizer, scheduler)
    cleanup()


def run_training_loop(rank, world_size, model, tokenizer, train_dataset, train_loader, test_loader,
                      train_sampler, num_epochs, start_epoch, checkpoint_dir,
                      optimizer, scheduler):
    device = torch.device(f'cuda:{rank}')
    criterion = TripletLoss(margin=0)
    accumulation_steps = 4

    is_main_process = (rank == 0)
    writer = SummaryWriter(log_dir="./runs/albertina_triplet") if is_main_process else None

    effective_batch_size = train_loader.batch_size * world_size * accumulation_steps
    if is_main_process:
        logger.info(f"Effective batch size: {effective_batch_size}")
        logger.info(f"Starting from epoch: {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        criterion.set_margin(min(0.3, ((epoch + 1) / 5) * 0.3))
        model.train()
        running_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()

        train_sampler.set_epoch(epoch)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") if is_main_process else train_loader

        for i, batch in enumerate(progress_bar):
            anchor_inputs = {k: v.to(device, non_blocking=True) for k, v in batch['anchor'].items()}
            positive_inputs = {k: v.to(device, non_blocking=True) for k, v in batch['positive'].items()}
            negative_inputs = {k: v.to(device, non_blocking=True) for k, v in batch['negative'].items()}

            should_sync = (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader)

            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                anchor_embeddings = model(anchor_inputs['input_ids'], anchor_inputs['attention_mask'])
                positive_embeddings = model(positive_inputs['input_ids'], positive_inputs['attention_mask'])
                negative_embeddings = model(negative_inputs['input_ids'], negative_inputs['attention_mask'])
                loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings) / accumulation_steps

            if world_size > 1 and not should_sync:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()

            if should_sync:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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

        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

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

            if test_loader is not None and (epoch + 1) % 10 == 0:
                model.eval()
                test_loss = 0.0
                test_batches = 0

                with torch.no_grad():
                    for batch in test_loader:
                        anchor_inputs = {k: v.to(device, non_blocking=True) for k, v in batch['anchor'].items()}
                        positive_inputs = {k: v.to(device, non_blocking=True) for k, v in batch['positive'].items()}
                        negative_inputs = {k: v.to(device, non_blocking=True) for k, v in batch['negative'].items()}

                        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                            anchor_embeddings = model(anchor_inputs['input_ids'], anchor_inputs['attention_mask'])
                            positive_embeddings = model(positive_inputs['input_ids'], positive_inputs['attention_mask'])
                            negative_embeddings = model(negative_inputs['input_ids'], negative_inputs['attention_mask'])
                            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)

                        test_loss += loss.item()
                        test_batches += 1

                avg_test_loss = test_loss / test_batches if test_batches > 0 else 0

                all_anchors = []
                all_positives = []

                model.eval()
                with torch.no_grad():
                    for batch in test_loader:
                        anchor_inputs = {k: v.to(device) for k, v in batch["anchor"].items()}
                        positive_inputs = {k: v.to(device) for k, v in batch["positive"].items()}

                        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                            anchor_embeds = model(anchor_inputs["input_ids"], anchor_inputs["attention_mask"])
                            positive_embeds = model(positive_inputs["input_ids"], positive_inputs["attention_mask"])

                        all_anchors.append(anchor_embeds)
                        all_positives.append(positive_embeds)

                anchors = torch.cat(all_anchors, dim=0)
                positives = torch.cat(all_positives, dim=0)

                similarities = torch.matmul(anchors, positives.T)
                ranks = similarities.argsort(dim=-1, descending=True)

                recall_at_5 = 0
                mrr_total = 0
                N = anchors.size(0)

                for i in range(N):
                    rank_list = ranks[i]
                    correct_index = i
                    rank_position = (rank_list == correct_index).nonzero(as_tuple=True)[0].item()
                    if rank_position < 5:
                        recall_at_5 += 1
                    mrr_total += 1.0 / (rank_position + 1)

                recall_at_5 /= N
                mrr_score = mrr_total / N

                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {avg_test_loss:.4f}, Recall@5: {recall_at_5:.4f}, "
                    f"MRR: {mrr_score:.4f}")

                if writer:
                    writer.add_scalar("Loss/Test", avg_test_loss, epoch)
                    writer.add_scalar("Metrics/Recall@5", recall_at_5, epoch)
                    writer.add_scalar("Metrics/MRR", mrr_score, epoch)

        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, epoch + 1, checkpoint_dir, rank)

        if rank == 0 and (epoch + 1) % 50 == 0:
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
