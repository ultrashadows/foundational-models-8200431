import os

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


# Setup distributed training to utilize all GPUs during training
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


# Cleanup dist after training
def cleanup():
    dist.destroy_process_group()


# Training function for each GPU
def train(rank, world_size, num_epochs=10):
    setup(rank, world_size)

    collection = connect().database.encoder

    model_name = "PORTULAN/albertina-1b5-portuguese-ptpt-encoder"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EncoderWrapper(model_name=model_name).to(rank)

    model = DDP(model, device_ids=[rank])

    dataset = DatasetProcessor(collection=collection, tokenizer=tokenizer)
    sampler = DistributedSampler(dataset=dataset, num_replicas=world_size, rank=rank)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=100,
        sampler=sampler,
        pin_memory=True
    )

    criterion = TripletLoss(margin=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        sampler.set_epoch(epoch)

        progress_bar = tqdm(data_loader) if rank == 0 else data_loader

        for batch in progress_bar:
            optimizer.zero_grad()

            anchor_inputs = {k: v.to(rank) for k, v in batch['anchor'].items()}
            positive_inputs = {k: v.to(rank) for k, v in batch['positive'].items()}
            negative_inputs = {k: v.to(rank) for k, v in batch['negative'].items()}

            anchor_embeddings = model(anchor_inputs['input_ids'], anchor_inputs['attention_mask'])
            positive_embeddings = model(positive_inputs['input_ids'], positive_inputs['attention_mask'])
            negative_embeddings = model(negative_inputs['input_ids'], negative_inputs['attention_mask'])

            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if rank == 0 and isinstance(progress_bar, tqdm):
                progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

        if rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {running_loss / len(data_loader):.4f}")

    if rank == 0:
        unwrapped_model = model.module.encoder
        unwrapped_model.save_pretrained("./finetuned-albertina-triplet-loss")
        tokenizer.save_pretrained("./finetuned-albertina-triplet-loss")
        print("Model saved successfully.")


def main():
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f'Found {world_size} GPUs. Training with distributed data parallel.')
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        print('Only one GPU found. Training on single GPU.')
        train(0, 1)


if __name__ == '__main__':
    main()
