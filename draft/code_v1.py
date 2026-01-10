import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import MultilabelF1Score

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
class Config:
    MAIN_DIR = "/home/shared/RD-Agent/online/cafa-6-protein-function-prediction"
    train_sequences_path = os.path.join(MAIN_DIR, "Train/train_sequences.fasta")
    train_labels_path = os.path.join(MAIN_DIR, "Train/train_terms.tsv")
    test_sequences_path = os.path.join(MAIN_DIR, "Test/testsuperset.fasta")

    embeddings_dir = os.path.expanduser("~/cafa6_embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    model_name = "esm2_t33_650M_UR50D"
    num_labels = 500
    n_epochs = 8
    batch_size = 128
    lr = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()


# -----------------------------------------------------------
# STEP 1: EXTRACT EMBEDDINGS USING FAIR-ESM
# -----------------------------------------------------------
from esm import FastaBatchedDataset, pretrained

def extract_embeddings(model_name, fasta_file, output_dir, repr_layers=[33], tokens_per_batch=4096, seq_length=1022):
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(seq_length),
        batch_sampler=batches,
    )

    os.makedirs(output_dir, exist_ok=True)
    print(f"Extracting embeddings to {output_dir}")

    all_embeds, all_ids = [], []

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing batch {batch_idx + 1}/{len(batches)}")
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)
            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {layer: t.to("cpu") for layer, t in out["representations"].items()}

            for i, label in enumerate(labels):
                entry_id = label.split()[0]
                truncate_len = min(seq_length, len(strs[i]))
                mean_embed = representations[repr_layers[0]][i, 1 : truncate_len + 1].mean(0).clone()
                all_embeds.append(mean_embed.numpy())
                all_ids.append(entry_id)

    np.save(os.path.join(output_dir, "embeddings.npy"), np.stack(all_embeds))
    np.save(os.path.join(output_dir, "ids.npy"), np.array(all_ids))
    print(f"Saved {len(all_ids)} embeddings to {output_dir}")


# -----------------------------------------------------------
# STEP 2: DATASET DEFINITION
# -----------------------------------------------------------
class ProteinSequenceDataset(Dataset):
    def __init__(self, datatype, embedding_dir):
        super().__init__()
        self.datatype = datatype
        embeds = np.load(os.path.join(embedding_dir, "embeddings.npy"))
        ids = np.load(os.path.join(embedding_dir, "ids.npy"))

        self.df = pd.DataFrame({"EntryID": ids, "embed": list(embeds)})

        if datatype == "train":
            np_labels = np.load(
                os.path.join(embedding_dir, f"train_targets_top{config.num_labels}.npy")
            )
            df_labels = pd.DataFrame({"EntryID": ids, "labels_vect": list(np_labels)})
            self.df = self.df.merge(df_labels, on="EntryID")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        embed = torch.tensor(self.df.iloc[index]["embed"], dtype=torch.float32)
        if self.datatype == "train":
            targets = torch.tensor(self.df.iloc[index]["labels_vect"], dtype=torch.float32)
            return embed, targets
        else:
            entry_id = self.df.iloc[index]["EntryID"]
            return embed, entry_id


# -----------------------------------------------------------
# STEP 3: MODELS
# -----------------------------------------------------------
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 864)
        self.linear2 = nn.Linear(864, 712)
        self.linear3 = nn.Linear(712, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        return x


class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 3, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(3, 8, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(int(8 * input_dim / 4), 864)
        self.fc2 = nn.Linear(864, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, embed_dim)
        x = self.pool1(torch.tanh(self.conv1(x)))
        x = self.pool2(torch.tanh(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------------------------------------
# STEP 4: TRAINING FUNCTION
# -----------------------------------------------------------
def train_model(embedding_dir, model_type="mlp", train_size=0.9):
    train_dataset = ProteinSequenceDataset("train", embedding_dir)
    train_len = int(len(train_dataset) * train_size)
    train_set, val_set = random_split(train_dataset, [train_len, len(train_dataset) - train_len])
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

    input_dim = train_dataset.df.iloc[0]["embed"].shape[0]
    model = (
        MultiLayerPerceptron(input_dim, config.num_labels)
        if model_type == "mlp"
        else CNN1D(input_dim, config.num_labels)
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=1)
    criterion = nn.BCEWithLogitsLoss()
    f1 = MultilabelF1Score(num_labels=config.num_labels).to(config.device)

    print("BEGIN TRAINING...")
    for epoch in range(config.n_epochs):
        model.train()
        train_losses, train_scores = [], []
        for embed, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.n_epochs}"):
            embed, targets = embed.to(config.device), targets.to(config.device)
            optimizer.zero_grad()
            preds = model(embed)
            loss = criterion(preds, targets)
            score = f1(torch.sigmoid(preds), targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_scores.append(score.item())

        model.eval()
        val_losses, val_scores = [], []
        with torch.no_grad():
            for embed, targets in val_loader:
                embed, targets = embed.to(config.device), targets.to(config.device)
                preds = model(embed)
                loss = criterion(preds, targets)
                score = f1(torch.sigmoid(preds), targets)
                val_losses.append(loss.item())
                val_scores.append(score.item())

        scheduler.step(np.mean(val_losses))
        print(f"Epoch {epoch+1}: Train F1={np.mean(train_scores):.4f}, Val F1={np.mean(val_scores):.4f}")

    print("TRAINING FINISHED ✅")
    return model


# -----------------------------------------------------------
# STEP 5: PREDICTION
# -----------------------------------------------------------
def predict(model, embedding_dir):
    test_dataset = ProteinSequenceDataset("test", embedding_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    labels = pd.read_csv(config.train_labels_path, sep="\t")
    top_terms = labels.groupby("term")["EntryID"].count().sort_values(ascending=False)
    labels_names = top_terms[:config.num_labels].index.values

    ids_, go_terms_, confs_ = [], [], []
    model.eval()
    with torch.no_grad():
        for embed, entry_id in tqdm(test_loader, desc="Predicting"):
            embed = embed.to(config.device)
            preds = torch.sigmoid(model(embed)).squeeze().cpu().numpy()
            ids_.extend([entry_id[0]] * config.num_labels)
            go_terms_.extend(labels_names)
            confs_.extend(preds.tolist())

    submission_df = pd.DataFrame({"Id": ids_, "GO term": go_terms_, "Confidence": confs_})
    submission_df.to_csv("submission.tsv", sep="\t", index=False, header=False)
    print("✅ Submission saved to submission.tsv")
    return submission_df


# -----------------------------------------------------------
# EXECUTION PIPELINE
# -----------------------------------------------------------
if __name__ == "__main__":
    # 1️⃣ Extract embeddings (can be skipped if already generated)
    extract_embeddings(
        model_name=config.model_name,
        fasta_file=config.train_sequences_path,
        output_dir=config.embeddings_dir,
    )

    # 2️⃣ Train model
    model = train_model(embedding_dir=config.embeddings_dir, model_type="mlp")

    # 3️⃣ Predict on test set
    submission_df = predict(model, config.embeddings_dir)
