import argparse

import torch
import torch.nn as nn
from datasets import load_dataset
from laser_encoders import LaserEncoderPipeline
from scipy.stats import pearsonr, spearmanr
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from tqdm import tqdm

import models.prediction_head as prediction_head
from utils import split_dataset, normalize


def main():
    # Handle arguments
    # ----------------
    parser = argparse.ArgumentParser(description="Train and score a model.")
    parser.add_argument("output_dir", type=str, help="Directory to save the best model.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="RicardoRei/wmt-da-human-evaluation",
        help="Name of the dataset to use.",
    )
    parser.add_argument("--epochs",
        type=int, default=10,
        help="Number of epochs to train the model for. Default is 10."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to print dataset structure and sample examples.",
    )
    args = parser.parse_args()

    # Load in the dataset
    # -------------------
    ds = load_dataset(args.dataset_name, split="train")
    ds = ds.filter(lambda col: col["lp"] == "de-en")
    ds = normalize(ds, "raw")  # Normalize the raw scores to [0, 1]
    columns_to_keep = ["src", "mt", "raw_norm"]
    ds = ds.remove_columns(
        [col for col in ds.column_names if col not in columns_to_keep]
    )
    # Filter out rows with any blank values
    initial_count = len(ds)
    ds = ds.filter(lambda row: all(row[col] for col in columns_to_keep))
    filtered_count = initial_count - len(ds)
    print(f"Filtered out {filtered_count} rows with blank values.")

    # TODO: we should filter to take the shortest sentences only

    # create training/validation/test split
    ds = split_dataset(ds)

    if args.debug:
        # Sanity-check: print dataset structure and a few examples
        print("🚨 Dataset split sizes:")
        for split in ["train", "validation", "test"]:
            print(f"{split}: {len(ds[split])} examples")

        print("\n🔍 Sample from training set:")
        sample = ds["train"][176]
        print("SRC:", sample["src"])
        print("MT:", sample["mt"])
        print("Normalized Score (raw_norm):", sample["raw_norm"])

    train_loader = DataLoader(
        ds["train"], batch_size=512, shuffle=True, num_workers=1, pin_memory=True
    )
    val_loader = DataLoader(ds["validation"], batch_size=512)

    # Initialize the encoders
    # -----------------------
    de_encoder = LaserEncoderPipeline(lang="deu_Latn")
    en_encoder = LaserEncoderPipeline(lang="eng_Latn")

    # Initialize the model and send to GPU
    # ------------------------------------
    model = prediction_head.PredictionHead()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get ready to train the model
    # ----------------------------
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.01)
    loss_fn = nn.MSELoss()
    patience = 2  # Early stopping patience
    best_val_loss = float("inf")
    stopping_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader):
            src = batch["src"]
            mt = batch["mt"]
            score = batch["raw_norm"]
            # encode sentences
            src_emb = de_encoder.encode_sentences(src)
            can_emb = en_encoder.encode_sentences(mt)
            score = score.float()

            # pass those puppies off to tensors
            src_emb = torch.tensor(src_emb, dtype=torch.float32)
            can_emb = torch.tensor(can_emb, dtype=torch.float32)

            src_emb, can_emb, score = (
                src_emb.to(device),
                can_emb.to(device),
                score.to(device),
            )

            optimizer.zero_grad()
            preds = model(src_emb, can_emb)
            loss = loss_fn(preds, score)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * src_emb.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader):
                src = batch["src"]
                mt = batch["mt"]
                score = batch["raw_norm"]
                src_emb = de_encoder.encode_sentences(src)
                can_emb = en_encoder.encode_sentences(mt)
                # pass those puppies off to tensors
                src_emb = torch.tensor(src_emb, dtype=torch.float32)
                can_emb = torch.tensor(can_emb, dtype=torch.float32)
                score = score.float()

                src_emb, can_emb, score = (
                    src_emb.to(device),
                    can_emb.to(device),
                    score.to(device),
                )

                preds = model(src_emb, can_emb)
                loss = loss_fn(preds, score)

                val_loss += loss.item() * src_emb.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(score.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        scheduler.step()

        pearson = pearsonr(all_preds, all_targets)[0]
        spearman = spearmanr(all_preds, all_targets)[0]

        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | r = {pearson:.4f}, p = {spearman:.4f}"
        )

        # Write epoch and loss stats to a file
        with open("training_log.txt", "a") as log_file:
            log_file.write(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | r = {pearson:.4f}, p = {spearman:.4f}\n"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stopping_counter = 0
            torch.save(model.state_dict(), args.output_dir + "/best_model.pt")
        else:
            stopping_counter += 1
            if stopping_counter >= patience:
                print("Early stopping.")
                break


if __name__ == "__main__":
    main()
