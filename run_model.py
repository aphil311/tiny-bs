import argparse

from laser_encoders import LaserEncoderPipeline

from models.prediction_head import PredictionHead
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Run the model to predict similarity score."
    )
    parser.add_argument("reference", type=str, help="reference sentence.")
    parser.add_argument("candidate", type=str, help="candidate sentence.")
    args = parser.parse_args()

    de_encoder = LaserEncoderPipeline(lang="deu_Latn")
    en_encoder = LaserEncoderPipeline(lang="eng_Latn")

    de_emb = de_encoder.encode_sentences([args.reference])
    en_emb = en_encoder.encode_sentences([args.candidate])

    de_emb = torch.tensor(de_emb)
    en_emb = torch.tensor(en_emb)

    prediction_head = PredictionHead()
    prediction_head.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        similarity_score = prediction_head(
            de_emb, en_emb
        )  # Get the similarity score as a Python float
    print(f"Similarity score: {similarity_score}")


if __name__ == "__main__":
    main()
