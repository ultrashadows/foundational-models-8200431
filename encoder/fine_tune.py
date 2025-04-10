import os
import platform
import logging

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
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


def setup(rank, world_size, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    logger.info(f"[RANK {rank}] Process group initialized using {backend} backend.")


def cleanup():
    dist.destroy_process_group()
    logger.info("Cleaned up distributed process group.")


def train_distributed(rank, world_size, num_epochs=10):
    backend = 'nccl' if platform.system() != 'Windows' else 'gloo'
    setup(rank, world_size, backend=backend)

    collection = connect().database.encoder

    model_name = "PORTULAN/albertina-1b5-portuguese-ptpt-encoder"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EncoderWrapper(model_name=model_name).to(rank)
    model = DDP(model, device_ids=[rank])

    logger.info(f"[RANK {rank}] Model loaded and wrapped in DDP.")

    train_dataset = DatasetProcessor(collection=collection, tokenizer=tokenizer, split="train", max_samples=1000)
    train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset=train_dataset, batch_size=10, sampler=train_sampler, pin_memory=torch.cuda.is_available())

    test_loader = None
    if rank == 0:
        test_dataset = DatasetProcessor(collection=collection, tokenizer=tokenizer, split="test")
        test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False, pin_memory=torch.cuda.is_available())

    run_training_loop(rank, world_size, model, tokenizer, train_loader, test_loader, train_sampler, num_epochs)
    cleanup()


def train_single_gpu(num_epochs=10):
    collection = connect().database.encoder

    model_name = "PORTULAN/albertina-1b5-portuguese-ptpt-encoder"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderWrapper(model_name=model_name).to(device)

    logger.info("Model loaded for single-GPU training.")

    train_dataset = DatasetProcessor(collection=collection, tokenizer=tokenizer, split="train", max_samples=1000)
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, pin_memory=torch.cuda.is_available())

    test_dataset = DatasetProcessor(collection=collection, tokenizer=tokenizer, split="test", max_samples=1000)
    test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, pin_memory=torch.cuda.is_available())

    run_training_loop(device, 1, model, tokenizer, train_loader, test_loader, None, num_epochs)


def run_training_loop(rank, world_size, model, tokenizer, train_loader, test_loader, train_sampler, num_epochs):
    device = rank if isinstance(rank, torch.device) else torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = TripletLoss(margin=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    is_main_process = (rank == 0 or world_size == 1)

    logger.info(f"[RANK {rank}] Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        if train_sampler:
            train_sampler.set_epoch(epoch)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") if is_main_process else train_loader

        for batch in progress_bar:
            optimizer.zero_grad()

            anchor_inputs = {k: v.to(device) for k, v in batch['anchor'].items()}
            positive_inputs = {k: v.to(device) for k, v in batch['positive'].items()}
            negative_inputs = {k: v.to(device) for k, v in batch['negative'].items()}

            anchor_embeddings = model(anchor_inputs['input_ids'], anchor_inputs['attention_mask'])
            positive_embeddings = model(positive_inputs['input_ids'], positive_inputs['attention_mask'])
            negative_embeddings = model(negative_inputs['input_ids'], negative_inputs['attention_mask'])

            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if is_main_process and isinstance(progress_bar, tqdm):
                progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
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

                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Test Avg Loss: {test_loss / len(test_loader):.4f}")

    if is_main_process:
        model_to_save = model.module.encoder if hasattr(model, "module") else model.encoder
        model_to_save.save_pretrained("./finetuned-albertina-triplet-loss")
        tokenizer.save_pretrained("./finetuned-albertina-triplet-loss")
        logger.info("Model and tokenizer saved successfully to ./finetuned-albertina-triplet-loss")


def main():
    setup_logging()
    world_size = torch.cuda.device_count()
    if world_size > 1 and platform.system() != "Windows":
        logger.info(f"Found {world_size} GPUs. Launching DDP training...")
        mp.spawn(train_distributed, args=(world_size,), nprocs=world_size, join=True)
    else:
        logger.info("Single GPU or Windows system detected. Running in single-GPU mode.")
        train_single_gpu()


if __name__ == '__main__':
    main()
