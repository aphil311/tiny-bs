import argparse

import torch
import torch.nn as nn
from datasets import load_dataset
from laser_encoders import LaserEncoderPipeline
from scipy.stats import pearsonr, spearmanr
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

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
        required=True,
        help="Name of the dataset to use.",
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

    # TODO: we should filter to take the shortest sentences only

    # create training/validation/test split
    ds = split_dataset(ds)

    train_loader = DataLoader(
        ds["train"], batch_size=512, shuffle=True, num_workers=4, pin_memory=True
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

    for epoch in range(10):
        model.train()
        train_loss = 0.0

        for src, mt, score in train_loader:
            # encode sentences
            src_emb = de_encoder.encode_sentences(src)
            can_emb = en_encoder.encode_sentences(mt)
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
            for src, mt, score in val_loader:
                src_emb = de_encoder.encode_sentences(src)
                can_emb = en_encoder.encode_sentences(mt)

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
        scheduler.step(val_loss)

        pearson = pearsonr(all_preds, all_targets)[0]
        spearman = spearmanr(all_preds, all_targets)[0]

        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | r = {pearson:.4f}, p = {spearman:.4f}"
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
