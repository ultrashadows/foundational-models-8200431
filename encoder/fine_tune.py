import os
import logging
import gc
import random
import warnings
import time
import psutil
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F

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
    optimizer_state = None
    scheduler_state = None

    if latest_checkpoint and os.path.exists(latest_checkpoint):
        if rank == 0:
            logger.info(f"Loading checkpoint from: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            optimizer_state = checkpoint.get('optimizer_state_dict', None)
            scheduler_state = checkpoint.get('scheduler_state_dict', None)
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

    return model, start_epoch, optimizer_state, scheduler_state


def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir, rank):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch
        }
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        cleanup_old_checkpoints(checkpoint_dir, keep_last=3)


def cleanup_old_checkpoints(checkpoint_dir, keep_last=3):
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


def train_distributed(rank, world_size, num_epochs=10, start_index=1, num_datasets=1):
    dist.init_process_group(backend='nccl')

    configure_gpu_memory()
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    base_model_path = "./models/albertina-1b5"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prev_checkpoint_path = "./models/albertina-1b5"

    for dataset_id in range(start_index, start_index + num_datasets):
        dataset_name = f"dataset_{dataset_id}.json"
        checkpoint_dir = f"./checkpoints/dataset_{dataset_id}"
        if rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)

        model, start_epoch, optimizer_state, scheduler_state = load_model_from_checkpoint_or_base(
            base_model_path=prev_checkpoint_path,
            checkpoint_dir=checkpoint_dir,
            device=device,
            rank=rank
        )

        prev_checkpoint_path = checkpoint_dir

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

        projection_params = list(model.module.projection.parameters()) if hasattr(model, "module") else list(
            model.projection.parameters())
        encoder_params = [p for n, p in model.named_parameters() if "projection" not in n]

        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "weight_decay": 0.01},
            {"params": projection_params, "weight_decay": 0.1}
        ], lr=1e-5, betas=(0.9, 0.999), eps=1e-8)

        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        train_dataset = DatasetProcessor(tokenizer, split="train", max_length=512, dataset=dataset_name)

        if start_epoch < 3:
            logger.info(f"Skipping hard negative precomputation for early epochs (epoch {start_epoch})")
        else:
            logger.info("Precomputing hard negatives")
            train_dataset.precompute_hard_negatives(model, tokenizer, device)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
                                           drop_last=True)
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, pin_memory=True,
                                  num_workers=32, persistent_workers=True, prefetch_factor=8, drop_last=True)

        test_loader = None
        if rank == 0:
            test_dataset = DatasetProcessor(tokenizer, split="test", max_length=512, dataset=dataset_name)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True,
                                     num_workers=16, persistent_workers=True, prefetch_factor=4, drop_last=False)

        total_steps = len(train_loader) * num_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=total_steps
        )

        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

        logger.info(f"Starting training on {dataset_name}")
        run_training_loop(rank, world_size, model, tokenizer, train_dataset, train_loader, test_loader,
                          train_sampler, num_epochs, start_epoch, checkpoint_dir, optimizer, scheduler, dataset_id)

        if dist.is_initialized():
            dist.barrier()
        torch.cuda.empty_cache()
        gc.collect()

    cleanup()


def run_training_loop(rank, world_size, model, tokenizer, train_dataset, train_loader, test_loader,
                      train_sampler, num_epochs, start_epoch, checkpoint_dir,
                      optimizer, scheduler, dataset_id):
    device = torch.device(f'cuda:{rank}')

    initial_margin = 0.5
    final_margin = 0.1
    criterion = TripletLoss(margin=initial_margin, mining_strategy='adaptive')

    accumulation_steps = 8
    max_grad_norm = 0.5

    is_main_process = (rank == 0)
    writer = SummaryWriter(log_dir=f"./runs/albertina_triplet_dataset_{dataset_id}") if is_main_process else None

    effective_batch_size = train_loader.batch_size * world_size * accumulation_steps
    if is_main_process:
        logger.info(f"Effective batch size: {effective_batch_size}")
        logger.info(f"Starting from epoch: {start_epoch}")

    best_loss = float('inf')
    patience_counter = 0
    patience = 8
    loss_history = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()

        progress = epoch / num_epochs
        current_margin = initial_margin * (1 - progress) + final_margin * progress
        criterion.set_margin(current_margin)

        train_sampler.set_epoch(epoch)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") if is_main_process else train_loader

        if epoch > 1 and epoch % 2 == 0 and hasattr(train_dataset, 'precompute_hard_negatives'):
            if is_main_process:
                logger.info(f"Recomputing hard negatives at epoch {epoch + 1}")
            train_dataset.precompute_hard_negatives(model, tokenizer, device)

        batch_losses = []

        for i, batch in enumerate(progress_bar):
            anchor_inputs = {k: v.to(device, non_blocking=True) for k, v in batch['anchor'].items()}
            positive_inputs = {k: v.to(device, non_blocking=True) for k, v in batch['positive'].items()}
            negative_inputs = {k: v.to(device, non_blocking=True) for k, v in batch['negative'].items()}
            weights = batch['weight'].to(device, non_blocking=True) if 'weight' in batch else None

            should_sync = (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader)

            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                anchor_embeddings = model(anchor_inputs['input_ids'], anchor_inputs['attention_mask'])
                positive_embeddings = model(positive_inputs['input_ids'], positive_inputs['attention_mask'])
                negative_embeddings = model(negative_inputs['input_ids'], negative_inputs['attention_mask'])

                if epoch > 3 and random.random() < 0.3:
                    all_embeddings = torch.cat([anchor_embeddings, positive_embeddings], dim=0)
                    batch_size = anchor_embeddings.size(0)
                    labels = torch.cat([
                        torch.arange(batch_size, device=device),
                        torch.arange(batch_size, device=device)
                    ])

                    pos_dist, neg_dist = criterion.batch_hard_mining(all_embeddings, labels)
                    pos_dist = pos_dist[:batch_size]
                    neg_dist = neg_dist[:batch_size]

                    hard_loss = F.relu(pos_dist - neg_dist + current_margin).mean()

                    regular_loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings, weights)
                    loss = 0.7 * regular_loss + 0.3 * hard_loss
                else:
                    loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings, weights)

                loss = loss / accumulation_steps

            if epoch > 2 and random.random() < 0.3:
                alpha = torch.rand(anchor_embeddings.size(0), 1, device=device)
                mixup = F.normalize(alpha * anchor_embeddings + (1 - alpha) * positive_embeddings, p=2, dim=-1)
                mixup_loss = criterion(mixup, positive_embeddings, negative_embeddings)
                loss = 0.7 * loss + 0.3 * mixup_loss

            if world_size > 1 and not should_sync:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()

            if should_sync:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

                if torch.isnan(grad_norm) or grad_norm > 10.0:
                    if is_main_process:
                        logger.warning(f"Large gradient norm detected: {grad_norm:.4f}, skipping step")
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            batch_loss = loss.item() * accumulation_steps
            running_loss += batch_loss
            batch_losses.append(batch_loss)
            num_batches += 1

            if is_main_process and isinstance(progress_bar, tqdm):
                current_lr = scheduler.get_last_lr()[0]
                recent_loss = sum(batch_losses[-10:]) / min(10, len(batch_losses))
                progress_bar.set_description(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f"Loss: {recent_loss:.4f}, "
                    f"LR: {current_lr:.2e}, "
                    f"Margin: {current_margin:.3f}"
                )

            if (i + 1) % 100 == 0:
                empty_cache()

        if len(train_loader) % accumulation_steps != 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            if not (torch.isnan(grad_norm) or grad_norm > 10.0):
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad()

        epoch_time = time.time() - epoch_start_time
        avg_train_loss = running_loss / num_batches if num_batches > 0 else 0

        loss_history.append(avg_train_loss)

        if len(loss_history) > 5:
            recent_trend = sum(loss_history[-3:]) / 3 - sum(loss_history[-6:-3]) / 3
            if recent_trend > 0.1 and epoch > 5:
                if is_main_process:
                    logger.warning(f"Loss trend increasing: {recent_trend:.4f}, reducing learning rate")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.8

        if avg_train_loss < best_loss - 0.001:
            best_loss = avg_train_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if is_main_process:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

        if writer:
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Learning_Rate", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("Time/Epoch", epoch_time, epoch)
            writer.add_scalar("Margin", current_margin, epoch)

        if is_main_process:
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}, Train Avg Loss: {avg_train_loss:.4f}, "
                f"Time: {epoch_time:.2f}s, Samples/sec: {len(train_dataset) / epoch_time:.1f}, "
                f"Margin: {current_margin:.3f}, Patience: {patience_counter}/{patience}")

            if test_loader is not None and (epoch + 1) % 3 == 0:
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
                recall_at_10 = 0
                mrr_total = 0
                N = anchors.size(0)

                for i in range(N):
                    rank_list = ranks[i]
                    correct_index = i
                    rank_position = (rank_list == correct_index).nonzero(as_tuple=True)[0].item()

                    if rank_position < 5:
                        recall_at_5 += 1
                    if rank_position < 10:
                        recall_at_10 += 1
                    mrr_total += 1.0 / (rank_position + 1)

                recall_at_5 /= N
                recall_at_10 /= N
                mrr_score = mrr_total / N

                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {avg_test_loss:.4f}, "
                    f"Recall@5: {recall_at_5:.4f}, Recall@10: {recall_at_10:.4f}, MRR: {mrr_score:.4f}")

                if writer:
                    writer.add_scalar("Loss/Test", avg_test_loss, epoch)
                    writer.add_scalar("Metrics/Recall@5", recall_at_5, epoch)
                    writer.add_scalar("Metrics/Recall@10", recall_at_10, epoch)
                    writer.add_scalar("Metrics/MRR", mrr_score, epoch)

        if (epoch + 1) % 2 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, checkpoint_dir, rank)

        if rank == 0 and (epoch + 1) % 20 == 0:
            empty_cache()

    if is_main_process:
        model_to_save = model.module.encoder if hasattr(model, "module") else model.encoder
        final_save_path = f"./results/finetuned-albertina-1b5-triplet-loss-dataset-{dataset_id}"
        os.makedirs(final_save_path, exist_ok=True)
        model_to_save.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        logger.info(f"✅ Model and tokenizer saved to {final_save_path}")

    if writer:
        writer.close()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ[
        "PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024,expandable_segments:True,roundup_power2_divisions:16"
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
