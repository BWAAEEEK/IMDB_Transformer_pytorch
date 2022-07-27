import argparse
import os
import time
import pickle
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

from model import Transformer
from dataset import CustomDataset
from trainer import Trainer

def fix_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--vocab_path", type=str, default=os.getcwd()+"/dataset/vocab")
    args.add_argument("--data_path", type=str, default=os.getcwd()+"/dataset/tokenized_data")
    args.add_argument("--output_path", type=str, default=os.getcwd()+"/output/")

    args.add_argument("--max_len", type=int, default=200, help="max length of sentence in data")

    args.add_argument("--hidden_size", type=int, default=32, help="hidden size for all hidden layers of the model")
    args.add_argument("--attn_heads", type=int, default=2, help="the number of attention heads")
    args.add_argument("--transformer_layers", type=int, default=1, help="the number of transformer layers")

    args.add_argument("--num_workers", type=int, default=5)
    args.add_argument("--batch_size", type=int, default=20)
    args.add_argument("--epochs", type=int, default=1000)

    args.add_argument("--learning_rate", type=float, default=0.001)
    args.add_argument("--betas", default=(0.9, 0.999))
    args.add_argument("--weight_decay", default=0)

    args.add_argument("--seed", type=int, default=42)

    args = args.parse_args()

    fix_seed(args.seed)

    print("\nLoading Vocab ...", args.vocab_path)
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)

    print("vocab size:", vocab["size"])

    print()
    print("\nLoading Dataset", args.data_path)
    with open(args.data_path, "rb") as f:
        data = pickle.load(f)
    print("the size of data:", len(data["label"]))

    print()
    print("\nLoading Custom Dataset")
    dataset = CustomDataset(data, vocab, args)

    print()
    print("\nCreating Data Loader")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    print()
    print("\nBuilding Transformer Model")
    model = Transformer(vocab["size"], args).cuda()

    print("\nCreating Model Trainer")
    trainer = Trainer(model, dataloader, args)

    time.sleep(0.5)
    print("\n--- Training Start ---")
    for epoch in range(args.epochs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        trainer.train(epoch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    trainer.save(args.epochs, args.output_path)
